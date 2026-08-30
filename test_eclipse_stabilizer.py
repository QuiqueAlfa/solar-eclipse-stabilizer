import builtins
import contextlib
import csv
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import eclipse_stabilizer as stabilizer
from eclipse_stabilizer_core import video as video_module
from eclipse_stabilizer_core import render as render_module


class OuterLimbDetectorTests(unittest.TestCase):
    def synthetic_crescent(self, center, angle_deg, radius=80):
        image = np.zeros((360, 240), np.uint8)
        c = tuple(np.rint(center).astype(int))
        cv2.circle(image, c, radius, 175, -1, cv2.LINE_AA)
        angle = np.deg2rad(angle_deg)
        moon_center = (
            int(round(center[0] + 0.72 * radius * np.cos(angle))),
            int(round(center[1] + 0.72 * radius * np.sin(angle))),
        )
        cv2.circle(image, moon_center, radius, 0, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(image, (3, 3), 0.6)

    def test_rotating_partial_eclipse_keeps_solar_center(self):
        expected = np.array([121.0, 180.0])
        errors = []
        for angle in range(0, 360, 30):
            image = self.synthetic_crescent(expected, angle)
            detection = stabilizer.detect_limb(image, 80.0, expected)
            self.assertGreater(detection["quality"], 0.15, angle)
            errors.append(np.linalg.norm(detection["center"] - expected))
        self.assertLess(max(errors), 1.25)

    def test_detector_follows_real_translation_not_crescent_centroid(self):
        positions = [np.array([90.0, 151.0]), np.array([143.0, 205.0]), np.array([111.0, 170.0])]
        for i, expected in enumerate(positions):
            image = self.synthetic_crescent(expected, 40 + i * 95, radius=68)
            detection = stabilizer.detect_limb(image, 68.0, None)
            self.assertLess(np.linalg.norm(detection["center"] - expected), 1.25)


class RadialMeasurementTests(unittest.TestCase):
    def synthetic_crescent(self, center, angle_deg, radius=80, size=(360, 240), peak=175):
        image = np.zeros(size, np.uint8)
        c = tuple(np.rint(center).astype(int))
        cv2.circle(image, c, radius, peak, -1, cv2.LINE_AA)
        angle = np.deg2rad(angle_deg)
        moon_center = (
            int(round(center[0] + 0.72 * radius * np.cos(angle))),
            int(round(center[1] + 0.72 * radius * np.sin(angle))),
        )
        cv2.circle(image, moon_center, radius, 0, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(image, (3, 3), 0.6)

    def test_radial_measurement_invariance_and_scales(self):
        expected = np.array([121.0, 180.0])
        # Rotation of the lunar bite must not move the measured solar center.
        for angle in range(0, 360, 30):
            m = stabilizer.measure_radial_limb(self.synthetic_crescent(expected, angle), expected, 80.0)
            self.assertTrue(m["measured"], angle)
            self.assertLess(np.linalg.norm(m["center"] - expected), 1.5, angle)
        # A known translation is followed, not the crescent centroid.
        for i, p in enumerate([np.array([90.0, 151.0]), np.array([143.0, 205.0]), np.array([111.0, 170.0])]):
            m = stabilizer.measure_radial_limb(self.synthetic_crescent(p, 40 + i * 95, radius=68), p, 68.0)
            self.assertTrue(m["measured"])
            self.assertLess(np.linalg.norm(m["center"] - p), 1.5)
        # Exposure variation must not switch branches.
        centers = []
        for peak in (175, 120, 80):
            m = stabilizer.measure_radial_limb(self.synthetic_crescent(expected, 60, peak=peak), expected, 80.0)
            self.assertTrue(m["measured"])
            centers.append(m["center"])
        self.assertLess(max(np.linalg.norm(c - centers[0]) for c in centers), 2.0)
        # Two analysis scales are equivalent once normalized.
        m_small = stabilizer.measure_radial_limb(self.synthetic_crescent(expected, 45, radius=80), expected, 80.0, scale=1.0)
        big = expected * 2.0
        m_big = stabilizer.measure_radial_limb(self.synthetic_crescent(big, 45, radius=160, size=(720, 480)), big, 160.0, scale=2.0)
        self.assertLess(np.linalg.norm(m_small["center"] - expected) / 1.0, 1.5)
        self.assertLess(np.linalg.norm(m_big["center"] - big) / 2.0, 1.5)

    def test_short_arc_without_evidence_is_not_measured(self):
        image = np.zeros((360, 240), np.uint8)
        cv2.ellipse(image, (121, 180), (80, 80), 0, -10, 10, 175, 6, cv2.LINE_AA)
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        m = stabilizer.measure_radial_limb(image, np.array([121.0, 180.0]), 80.0)
        self.assertFalse(m["measured"])
        self.assertLess(m["valid_points"], 35)
        self.assertGreater(m["condition"], 1.0)
        self.assertGreaterEqual(m["largest_gap_deg"], 180.0)


class ProfileDiscardTests(unittest.TestCase):
    def test_without_profile_preserves_all_frames(self):
        frame_count = 2400
        keep = np.ones(frame_count, dtype=bool)

        selected_keep, discarded, messages = stabilizer.apply_profile_discards(keep)

        self.assertFalse(discarded.any())
        self.assertTrue(selected_keep.all())
        self.assertEqual(messages, [])

    def test_profile_applies_only_declared_ranges(self):
        frame_count = 2400
        keep = np.ones(frame_count, dtype=bool)
        profile = {
            "version": 1,
            "discards": [(20, 24), (700, 700), (900, 900)],
        }

        selected_keep, discarded, messages = stabilizer.apply_profile_discards(
            keep, profile=profile
        )

        self.assertTrue(selected_keep[19])
        self.assertFalse(selected_keep[20:25].any())
        self.assertTrue(selected_keep[25])
        self.assertTrue(selected_keep[699])
        self.assertFalse(selected_keep[700])
        self.assertTrue(selected_keep[701])
        self.assertTrue(selected_keep[899])
        self.assertFalse(selected_keep[900])
        self.assertTrue(selected_keep[901])
        self.assertTrue(discarded[20:25].all())
        self.assertTrue(discarded[700])
        self.assertTrue(discarded[900])
        self.assertFalse(discarded[25])
        self.assertFalse(discarded[701])
        self.assertTrue(any("20-24" in message for message in messages))

    def test_profile_with_no_declared_ranges_preserves_everything(self):
        frame_count = 200
        keep = np.ones(frame_count, dtype=bool)
        profile = {
            "version": 1,
            "discards": [],
        }

        selected_keep, discarded, _ = stabilizer.apply_profile_discards(
            keep, profile=profile
        )

        self.assertTrue(selected_keep.all())
        self.assertFalse(discarded.any())

    def test_profile_range_beyond_video_does_not_discard_last_frame(self):
        keep = np.ones(20, dtype=bool)
        profile = {
            "version": 1,
            "discards": [(100, 110)],
        }

        selected_keep, discarded, _ = stabilizer.apply_profile_discards(
            keep, profile=profile
        )

        self.assertTrue(selected_keep.all())
        self.assertFalse(discarded.any())

    def _minimal_analysis(self):
        n = 300
        raw_center = np.full((n, 2), np.nan, np.float64)
        raw_center[:100] = 0.0
        quality = np.zeros(n, np.float64)
        quality[:100] = 0.5
        coverage = np.zeros(n, np.float64)
        coverage[:100] = 360.0
        median_residual = np.full(n, 10.0, np.float64)
        median_residual[:100] = 0.5
        threshold = np.zeros(n, np.float64)
        touch = np.zeros(n, bool)
        radial_points = np.full(n, 180, np.int16)
        radial_strength = np.zeros(n, np.float64)
        x = np.arange(n, dtype=float)
        visible_center = np.column_stack(
            (0.1 * x + 2.0 * (-1.0) ** x, np.zeros(n, dtype=float))
        )
        relative = np.zeros((n, 2), np.float64)
        response = np.ones(n, np.float64)
        maximum = np.full(n, 200, np.uint8)
        contrast_center = np.full((n, 2), np.nan, np.float64)
        contrast_center[240:] = 0.0
        return {
            "raw_center": raw_center,
            "quality": quality,
            "coverage": coverage,
            "median_residual": median_residual,
            "threshold": threshold,
            "touch": touch,
            "radial_points": radial_points,
            "radial_strength": radial_strength,
            "visible_center": visible_center,
            "relative": relative,
            "response": response,
            "maximum": maximum,
            "contrast_center": contrast_center,
            "radius": np.array([20.0]),
            "analysis_width": np.array([270]),
            "analysis_height": np.array([480]),
            "source_frames": np.array([n]),
            "source_fps": np.array([30.0]),
        }

    def test_solve_tracking_applies_profile_discards_only_when_declared(self):
        analysis = self._minimal_analysis()
        base = stabilizer.solve_tracking(analysis, min_quality=0.18)
        self.assertFalse(base["timed_discarded"].any())

        discard_profile = {
            "version": 1,
            "discards": [(120, 122)],
        }
        with_discards = stabilizer.solve_tracking(analysis, 0.18, discard_profile)
        self.assertTrue(with_discards["timed_discarded"][120:123].all())
        self.assertFalse(with_discards["timed_discarded"][123:].any())
        self.assertFalse(with_discards["keep"][120:123].any())
        np.testing.assert_array_equal(with_discards["center"], base["center"])

    def test_load_profile_rejects_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaises(SystemExit):
                stabilizer.load_profile(str(missing))
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"version": 1, "discards": [{"frame": -3}]}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                stabilizer.load_profile(str(bad))
            for removed in ("visible_smoothing", "excursion_repair"):
                bad.write_text(json.dumps({"version": 1, removed: {"start": 0, "end": 10}}), encoding="utf-8")
                with self.assertRaises(SystemExit):
                    stabilizer.load_profile(str(bad))

    def test_load_profile_accepts_example_profile(self):
        profile_path = Path(__file__).resolve().parent / "profiles" / "example.json"
        profile = stabilizer.load_profile(str(profile_path))
        self.assertIsNotNone(profile)
        self.assertEqual(profile["discards"], [(120, 124), (900, 900)])
        self.assertNotIn("visible_smoothing", profile)
        self.assertNotIn("excursion_repair", profile)

    def test_cli_does_not_offer_square_mode(self):
        help_text = stabilizer.build_parser().format_help()
        self.assertNotIn("square", help_text)
        self.assertNotIn("cuadrado", help_text)
        self.assertNotIn("square-preview", help_text)
        self.assertNotIn("square-export", help_text)
        source = inspect.getsource(stabilizer)
        self.assertNotIn("square_crop_parameters", source)
        self.assertNotIn("square-preview", source)
        self.assertNotIn("square-export", source)


def render_crescent(shape, sun_center, moon_center, radius, intensity=175):
    image = np.zeros(shape, np.uint8)
    cv2.circle(image, tuple(np.rint(sun_center).astype(int)), int(radius), intensity, -1, cv2.LINE_AA)
    cv2.circle(image, tuple(np.rint(moon_center).astype(int)), int(radius), 0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0.6)


def bright_centroid(gray):
    ys, xs = np.nonzero(gray > 100)
    if len(xs) == 0:
        return np.array([np.nan, np.nan])
    return np.array([float(xs.mean()), float(ys.mean())])


def write_synthetic_video(tmp, n, width=240, height=360, fps=30.0, prefix="synth"):
    """Write a small synthetic video used for the Phase 4 export integration test."""
    path = Path(tmp) / f"{prefix}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("no se pudo crear el vídeo sintético")
    try:
        for _ in range(n):
            img = np.zeros((height, width, 3), np.uint8)
            cv2.circle(img, (width // 2, height // 2), 28, (200, 200, 200), -1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
    return path


def synthetic_analysis(n, radius=80.0, fps=30.0):
    """A clean full-circle analysis used as the base for state-policy tests."""
    raw_center = np.full((n, 2), np.nan, np.float64)
    raw_center[:] = [120.0, 180.0]
    quality = np.full(n, 0.5, np.float64)
    coverage = np.full(n, 360.0, np.float64)
    median_residual = np.full(n, 0.5, np.float64)
    threshold = np.full(n, 12.0, np.float64)
    touch = np.zeros(n, bool)
    radial_points = np.full(n, 180, np.int16)
    radial_strength = np.full(n, 10.0, np.float64)
    visible_center = np.column_stack((np.linspace(0, 10, n), np.zeros(n)))
    relative = np.zeros((n, 2), np.float64)
    response = np.ones(n, np.float64)
    maximum = np.full(n, 200, np.uint8)
    contrast_center = np.full((n, 2), np.nan, np.float64)
    contrast_score = np.zeros(n, np.float64)
    return {
        "raw_center": raw_center,
        "quality": quality,
        "coverage": coverage,
        "median_residual": median_residual,
        "threshold": threshold,
        "touch": touch,
        "radial_points": radial_points,
        "radial_strength": radial_strength,
        "visible_center": visible_center,
        "relative": relative,
        "response": response,
        "maximum": maximum,
        "contrast_center": contrast_center,
        "contrast_score": contrast_score,
        "radius": np.array([radius]),
        "analysis_width": np.array([270]),
        "analysis_height": np.array([480]),
        "source_frames": np.array([n]),
        "source_fps": np.array([fps]),
        "clip_edges": np.zeros(n, np.int8),
        "clip_score": np.zeros(n, np.float64),
    }


def render_tear_eclipse(
    angle_deg=0.0,
    moon_offset_r=0.80,
    tear=False,
    brightness=220.0,
    noise_sigma=0.0,
    seed=7,
):
    """Synthetic crescent plus an optional readout-failure half-plane."""
    shape = (480, 270)
    center = np.array([135.0, 240.0])
    radius = 100.0
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    relative = np.stack((xx - center[0], yy - center[1]), axis=-1)
    angle = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    image = np.full(shape, 2.0, np.float64)
    image[np.linalg.norm(relative, axis=2) <= radius] = brightness
    moon_center = center + moon_offset_r * radius * direction
    moon = np.hypot(xx - moon_center[0], yy - moon_center[1]) <= radius
    image[moon] = 2.0
    if tear:
        projection = relative @ (-direction)
        image[projection >= 0.38 * radius] = 0.0
    if noise_sigma > 0:
        image += np.random.default_rng(seed).normal(0.0, noise_sigma, shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), center, radius


class SensorTearDetectorTests(unittest.TestCase):
    def test_exposed_fraction_measures_the_missing_content(self):
        clean, center, radius = render_tear_eclipse(tear=False)
        torn, _, _ = render_tear_eclipse(tear=True)
        clean_result = stabilizer.measure_exposed_fraction(clean, center, radius)
        torn_result = stabilizer.measure_exposed_fraction(torn, center, radius)
        self.assertTrue(clean_result["evaluable"])
        self.assertTrue(torn_result["evaluable"])
        self.assertGreater(
            clean_result["exposed_fraction"] - torn_result["exposed_fraction"],
            0.15,
        )

    def test_exposed_fraction_is_stable_across_brightness(self):
        values = []
        for brightness in (220.0, 140.0, 80.0):
            gray, center, radius = render_tear_eclipse(brightness=brightness)
            result = stabilizer.measure_exposed_fraction(gray, center, radius)
            self.assertTrue(result["evaluable"])
            values.append(result["exposed_fraction"])
        self.assertLess(max(values) - min(values), 0.01)

    def test_underexposed_noise_abstains(self):
        gray, center, radius = render_tear_eclipse(
            brightness=14.0, noise_sigma=3.0
        )
        result = stabilizer.measure_exposed_fraction(gray, center, radius)
        self.assertFalse(result["evaluable"], result)

    def test_temporal_drop_is_detected_but_smooth_evolution_is_not(self):
        exposed = np.ones(61, np.float64)
        exposed[30] = 0.75
        result = stabilizer.detect_transient_exposure_drops(
            exposed, np.ones(61, bool), np.ones(61, bool), fps=30.0
        )
        self.assertEqual(np.flatnonzero(result["detected"]).tolist(), [30])
        smooth = np.linspace(1.0, 0.4, 61)
        smooth_result = stabilizer.detect_transient_exposure_drops(
            smooth, np.ones(61, bool), np.ones(61, bool), fps=30.0
        )
        self.assertFalse(smooth_result["candidate"].any())
        self.assertFalse(smooth_result["detected"].any())

    def test_temporal_duration_and_eligibility(self):
        expected = {24.0: 1, 25.0: 1, 30.0: 2, 60.0: 4, 120.0: 8}
        for fps, maximum in expected.items():
            self.assertEqual(maximum, int(np.floor(0.067 * fps)))
            # At 24/25 FPS the permitted maximum is a single frame, so no
            # positive run lies immediately below the 0.067 s limit.
            lengths = [("equal", maximum), ("above", maximum + 1)]
            if maximum > 1:
                lengths.insert(0, ("below", maximum - 1))
            for label, length in lengths:
                with self.subTest(fps=fps, run=label):
                    detected, _ = stabilizer.isolate_transient_tears(
                        np.ones(length, bool), fps
                    )
                    if label == "above":
                        self.assertFalse(detected.any())
                    else:
                        self.assertTrue(detected.all())

    def test_ineligible_frames_split_runs_like_clipping_or_horizon(self):
        fps = 30.0
        cases = {
            "clip": (np.array([True, True, False, True, True]), [(0, 2), (3, 5)]),
            "horizon": (np.array([True, True, False, False, False]), [(0, 2)]),
        }
        for reason, (eligible, segments) in cases.items():
            with self.subTest(reason=reason):
                detected, run_length = stabilizer.isolate_transient_tears(
                    np.ones(5, bool), fps, eligible=eligible
                )
                for start, stop in segments:
                    self.assertTrue(detected[start:stop].all())
                    self.assertEqual(int(run_length[start]), stop - start)
                self.assertFalse(detected[~np.asarray(eligible)].any())
                joined, _ = stabilizer.isolate_transient_tears(np.ones(5, bool), fps)
                self.assertFalse(joined.any())

    def test_content_mapping_clip_veto_and_manual_recovery(self):
        analysis = synthetic_analysis(61)
        analysis["tear_evaluable"] = np.ones(61, bool)
        analysis["tear_exposed_fraction"] = np.ones(61, np.float64)
        analysis["tear_exposed_fraction"][30] = 0.75
        solved = stabilizer.solve_tracking(analysis, min_quality=0.18)
        self.assertTrue(solved["tear_detected"][30])
        self.assertEqual(int(solved["content_state"][30]), stabilizer.CONTENT_CLIPPED)
        self.assertFalse(solved["keep"][30])
        combined = dict(analysis)
        combined.update(solved)
        selection = stabilizer.effective_selection(combined, keep="30")
        self.assertTrue(selection["mask"][30])
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "tracking.csv"
            stabilizer.write_tracking_csv(audit_path, combined)
            with audit_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertIn("tear_detected", rows[0])
            self.assertIn("tear_veto_reason", rows[0])
            self.assertEqual(len(rows[0]), len(rows[1]))

        clipped = synthetic_analysis(61)
        clipped["tear_evaluable"] = np.ones(61, bool)
        clipped["tear_exposed_fraction"] = np.ones(61, np.float64)
        clipped["tear_exposed_fraction"][30] = 0.75
        clipped["clip_edges"][30] = stabilizer.EDGE_BITS["right"]
        clipped["clip_score"][30] = 1.0
        vetoed = stabilizer.solve_tracking(clipped, min_quality=0.18)
        self.assertFalse(vetoed["tear_detected"][30])
        self.assertTrue(vetoed["border_clipped"][30])
        self.assertIn("classify_clipping", vetoed["tear_veto_reason"][30])


class GeneralizationPhaseTwoTests(unittest.TestCase):
    def _degrade_geometry(self, analysis, start, stop):
        analysis["quality"][start:stop] = 0.0
        analysis["coverage"][start:stop] = 60.0
        analysis["median_residual"][start:stop] = 8.0
        analysis["radial_points"][start:stop] = 5
        analysis["raw_center"][start:stop] = np.nan

    def _add_monotonic_horizon(self, analysis, start, sustain=12):
        stop = start + sustain
        analysis["quality"][start:] = 0.0
        analysis["coverage"][start:stop] = np.linspace(145.0, 35.0, sustain)
        analysis["coverage"][stop:] = 35.0
        analysis["median_residual"][start:stop] = np.linspace(4.6, 9.0, sustain)
        analysis["median_residual"][stop:] = 9.0
        analysis["radial_points"][start:stop] = np.rint(
            np.linspace(32, 3, sustain)
        ).astype(np.int16)
        analysis["radial_points"][stop:] = 3
        analysis["raw_center"][start:] = np.nan

    def test_real_clipping_is_detected_at_each_edge(self):
        radius = 80.0
        cases = {
            "left": (np.array([-30.0, 180.0]), np.array([-87.0, 180.0]), "left"),
            "right": (np.array([270.0, 180.0]), np.array([327.0, 180.0]), "right"),
            "top": (np.array([120.0, -30.0]), np.array([120.0, -87.0]), "top"),
            "bottom": (np.array([120.0, 390.0]), np.array([120.0, 447.0]), "bottom"),
        }
        for name, (sun, moon, edge) in cases.items():
            gray = render_crescent((360, 240), sun, moon, radius)
            predicted = bright_centroid(gray)
            self.assertTrue(np.isfinite(predicted).all(), name)
            result = stabilizer.classify_clipping(gray, predicted, radius)
            self.assertTrue(result["clipped"], name)
            self.assertIn(edge, result["edges"], name)
            self.assertIn("recorte", result["reason"], name)

    def test_theoretical_circle_outside_but_visible_crescent_inside(self):
        # Sun center is 40 px from the left border; the 80 px circle would be
        # cut by the theoretical criterion, but the Moon hides that side and
        # the visible crescent (x in [40,120]) is fully inside the canvas.
        gray = render_crescent((360, 240), np.array([40.0, 180.0]), np.array([-17.6, 180.0]), 80.0)
        predicted = bright_centroid(gray)
        result = stabilizer.classify_clipping(gray, predicted, 80.0)
        self.assertFalse(result["clipped"])
        self.assertEqual(result["edges"], [])
        # The theoretical circle does leave the canvas: cut_geometry-like
        # condition is satisfied, yet the content state must stay usable.
        margin = min(40.0 - 80.0, 180.0 - 80.0, (240 - 1.0) - (40.0 + 80.0), (360 - 1.0) - (180.0 + 80.0))
        self.assertLess(margin, -0.75)
        analysis = synthetic_analysis(30)
        analysis["clip_edges"][0] = 0
        states, reasons = stabilizer.classify_content(analysis, np.array([False] * 30), analysis["clip_edges"], analysis["clip_score"], horizon_start=30)
        self.assertEqual(int(states[0]), stabilizer.CONTENT_USABLE)
        self.assertIn("dentro del lienzo", reasons[0])

    def test_low_confidence_becomes_uncertain_but_is_kept_if_centrable(self):
        analysis = synthetic_analysis(30)
        analysis["radial_points"][10] = 5
        analysis["quality"][10] = 0.01
        solved = stabilizer.solve_tracking(analysis, min_quality=0.18)
        self.assertEqual(int(solved["content_state"][10]), stabilizer.CONTENT_UNCERTAIN)
        self.assertNotEqual(int(solved["centering_state"][10]), stabilizer.CENTER_UNRESOLVED)
        self.assertTrue(solved["keep"][10])
        self.assertNotIn("recorte", solved["content_reason"][10])

    def test_noise_or_bright_landscape_at_edge_is_not_confused_with_the_eclipse(self):
        # Sun fully inside plus a disconnected bright band at the bottom border
        # and isolated bright pixels on the left/top borders.
        gray = render_crescent((360, 240), np.array([120.0, 180.0]), np.array([62.4, 180.0]), 80.0)
        gray[330:360, :] = np.maximum(gray[330:360, :], 150)
        gray[0:2, 0:2] = 255
        gray[0:2, 238:240] = 255
        predicted = bright_centroid(gray)
        result = stabilizer.classify_clipping(gray, predicted, 80.0)
        self.assertFalse(result["clipped"])
        self.assertEqual(result["edges"], [])

    def test_two_scales_produce_equivalent_normalized_centers(self):
        gray_small = render_crescent((360, 240), np.array([121.0, 180.0]), np.array([63.4, 180.0]), 80.0)
        gray_large = render_crescent((720, 480), np.array([242.0, 360.0]), np.array([126.8, 360.0]), 160.0)
        small = stabilizer.detect_limb(gray_small, 80.0, None)
        large = stabilizer.detect_limb(gray_large, 160.0, None, scale=2.0)
        self.assertGreater(small["quality"], 0.15)
        self.assertGreater(large["quality"], 0.15)
        normalized_small = small["center"] / np.array([240.0, 360.0])
        normalized_large = large["center"] / np.array([480.0, 720.0])
        np.testing.assert_allclose(normalized_large, normalized_small, atol=0.01)

    def test_centering_states_and_unresolved_are_not_exportable(self):
        anchor = np.zeros(60, bool)
        anchor[0:10] = True
        anchor[50:60] = True
        states, reasons = stabilizer.classify_centering(anchor, source_fps=30.0)
        self.assertTrue((states[0:10] == stabilizer.CENTER_RELIABLE).all())
        # Interpolation between the two anchor blocks is reconstructible.
        self.assertTrue((states[10:50] == stabilizer.CENTER_RECONSTRUCTED).all())
        # Dead-end region with no neighbor anchor within the extrapolation window
        # (1 s = 30 frames) stays unresolved.
        dead = np.zeros(40, bool)
        dead[0:5] = True
        dead_states, dead_reasons = stabilizer.classify_centering(dead, source_fps=30.0)
        self.assertTrue((dead_states[5:35] == stabilizer.CENTER_RECONSTRUCTED).all())
        self.assertTrue((dead_states[35:] == stabilizer.CENTER_UNRESOLVED).all())

        # End-to-end policy: unresolved frames must never be exported.  Only the
        # first ten frames carry a direct measurement; the rest are a long
        # content-bearing stretch with no neighbor anchor beyond 1 second.
        analysis = synthetic_analysis(60)
        analysis["raw_center"][10:60] = np.nan
        analysis["radial_points"][10:60] = 0
        analysis["maximum"][10:60] = 200
        analysis["contrast_center"][10:60] = np.nan
        solved = stabilizer.solve_tracking(analysis, min_quality=0.18)
        self.assertTrue(solved["keep"][0:10].all())
        self.assertTrue(solved["keep"][10:40].all())
        self.assertFalse(solved["keep"][40:60].any())
        unresolved = solved["centering_state"] == stabilizer.CENTER_UNRESOLVED
        self.assertTrue(unresolved[40:60].all())
        # No unresolved frame is exported.
        self.assertFalse((solved["keep"] & unresolved).any())

    def test_transient_degradation_recovers_limbo(self):
        analysis = synthetic_analysis(60)
        self._degrade_geometry(analysis, 20, 27)

        result = stabilizer.classify_regime(analysis, sustain=12, recover=4)

        self.assertEqual(result["horizon_start"], 60)
        self.assertTrue(
            (result["regime"][20:27] == stabilizer.REGIME_TRANSIENT).all()
        )
        self.assertTrue((result["regime"][27:] == stabilizer.REGIME_LIMBO).all())

    def test_ambiguous_cloud_tail_does_not_confirm_horizon(self):
        analysis = synthetic_analysis(70)
        self._degrade_geometry(analysis, 35, 70)
        # Persistent but flat degradation: no monotonic physical loss.
        analysis["maximum"][35:] = 160
        analysis["touch"][35:] = False

        result = stabilizer.classify_regime(analysis, sustain=12, recover=4)

        self.assertEqual(result["horizon_start"], 70)
        self.assertTrue(
            (result["regime"][35:] == stabilizer.REGIME_TRANSIENT).all()
        )

    def test_confirmed_horizon_latches_and_rejects_later_circles(self):
        analysis = synthetic_analysis(75)
        self._add_monotonic_horizon(analysis, 30, sustain=12)
        # Artificially precise circles after physical horizon confirmation.
        analysis["raw_center"][58:64] = [120.0, 180.0]
        analysis["quality"][58:64] = 0.8
        analysis["coverage"][58:64] = 300.0
        analysis["median_residual"][58:64] = 0.2
        analysis["radial_points"][58:64] = 180

        result = stabilizer.classify_regime(analysis, sustain=12, recover=4)

        self.assertEqual(result["horizon_start"], 30)
        self.assertTrue(
            (result["regime"][30:] == stabilizer.REGIME_HORIZON).all()
        )
        self.assertTrue(result["false_circle_after_horizon"][58:64].all())
        solved = stabilizer.solve_tracking(analysis, min_quality=0.18)
        self.assertFalse(solved["geometry_trusted"][58:64].any())
        self.assertTrue(solved["false_circle_after_horizon"][58:64].all())

    def test_horizon_tail_is_framewise_not_globally_truncated(self):
        analysis = synthetic_analysis(70)
        self._add_monotonic_horizon(analysis, 30, sustain=12)
        analysis["contrast_center"][30:] = [120.0, 180.0]
        analysis["contrast_score"][30:] = 0.8
        analysis["clip_edges"][52] = stabilizer.EDGE_BITS["left"]
        analysis["clip_score"][52] = 0.8

        solved = stabilizer.solve_tracking(analysis, min_quality=0.18)

        self.assertEqual(int(solved["trim_end"][0]), 69)
        self.assertTrue(solved["keep"][30:52].all())
        self.assertFalse(solved["keep"][52])
        self.assertTrue(solved["keep"][53:].all())

    def test_soft_ramp_then_hard_collapse_confirms_horizon(self):
        # Real degradation profile: a soft one-signal ramp (arc still partially
        # measurable but disagreeing: coverage under threshold, radius-scale and
        # arc-shape loss, SRC_RADIAL_DISAGREE) followed by a total geometry
        # collapse with a persistent bright glow that never darkens.  The
        # radial-aware signals must confirm the horizon at the ramp start.
        n = 75
        analysis = synthetic_analysis(n)
        radius = float(analysis["radius"][0])
        analysis["geometry_source"] = np.full(n, stabilizer.SRC_RADIAL, np.int8)
        analysis["radius_meas"] = np.full(n, radius, np.float64)
        analysis["arc_measured"] = np.ones(n, bool)
        analysis["arc_gap_deg"] = np.full(n, 45.0, np.float32)
        # soft ramp 30..54: only coverage + radius-scale + arc-shape fail
        analysis["coverage"][30:55] = 135.0
        analysis["quality"][30:55] = 0.2
        analysis["median_residual"][30:55] = 1.5
        analysis["radial_points"][30:55] = 120
        analysis["radius_meas"][30:55] = 0.875 * radius
        analysis["arc_gap_deg"][30:55] = 225.0
        analysis["geometry_source"][30:55] = stabilizer.SRC_RADIAL_DISAGREE
        # hard collapse 55..end: total loss, glow persists (max >= 8)
        analysis["quality"][55:] = 0.0
        analysis["coverage"][55:] = 0.0
        analysis["median_residual"][55:] = np.inf
        analysis["radial_points"][55:] = 0
        analysis["radius_meas"][55:] = np.nan
        analysis["arc_measured"][55:] = False
        analysis["arc_gap_deg"][55:] = 360.0
        analysis["geometry_source"][55:] = stabilizer.SRC_NONE
        analysis["raw_center"][55:] = np.nan
        analysis["maximum"][30:] = 160

        result = stabilizer.classify_regime(analysis, sustain=12, recover=4)

        self.assertEqual(result["horizon_start"], 30)
        self.assertTrue((result["regime"][:30] == stabilizer.REGIME_LIMBO).all())
        self.assertTrue((result["regime"][30:] == stabilizer.REGIME_HORIZON).all())

    def test_fallback_does_not_reanchor_after_confirmed_horizon(self):
        # Post-horizon false circles with finite geometry must NOT re-anchor the
        # anchored contrast fallback when the accepted-trusted re-anchor mask is
        # passed; without the mask (legacy) they re-anchor and yank the track.
        n = 75
        analysis = synthetic_analysis(n)
        self._add_monotonic_horizon(analysis, 30, sustain=12)
        analysis["geometry_source"] = np.zeros(n, np.int8)
        analysis["geometry_source"][:30] = stabilizer.SRC_RADIAL
        analysis["radius_meas"] = np.full(n, 80.0)
        analysis["arc_measured"] = np.ones(n, bool)
        analysis["arc_gap_deg"] = np.full(n, 45.0)
        # false circular reappearance after the irreversible horizon latch
        analysis["raw_center"][58:64] = [100.0, 220.0]
        analysis["quality"][58:64] = 0.8
        analysis["coverage"][58:64] = 300.0
        analysis["median_residual"][58:64] = 0.2
        analysis["radial_points"][58:64] = 180
        analysis["geometry_source"][58:64] = stabilizer.SRC_RADIAL

        frames = [np.zeros((480, 270), np.uint8) for _ in range(n)]
        # Legacy behavior without a mask: any accepted geometry re-anchors.
        legacy = stabilizer.refresh_contrast_track_frames(analysis, frames)
        self.assertTrue(legacy["fallback_reanchored"][58:64].all())

        # Production semantics: accepted-trusted mask (limbo before the latch,
        # cleaned by the transient-outlier gate).
        regime_info = stabilizer.classify_regime(analysis, sustain=12, recover=4)
        geometry_usable = (
            regime_info["limbo_evidence"]
            & ~regime_info["false_circle_after_horizon"]
            & (np.arange(n) < regime_info["horizon_start"])
        )
        mask = stabilizer.transient_outliers(
            analysis["raw_center"], analysis["relative"], geometry_usable, gate=4.0
        )
        self.assertFalse(mask[58:64].any())
        masked = stabilizer.refresh_contrast_track_frames(analysis, frames, reanchor_mask=mask)
        self.assertFalse(masked["fallback_reanchored"][58:64].any())
        # The masked fallback propagates smoothly and never jumps to the false circle.
        np.testing.assert_allclose(masked["contrast_center"][58], masked["contrast_center"][57], atol=1e-9)
        self.assertFalse(np.allclose(masked["contrast_center"][58], [100.0, 220.0]))

    def test_unconfirmed_measurement_excursion_is_robustly_reduced(self):
        n = 100
        anchors = np.tile([100.0, 180.0], (n, 1))
        anchors[30:71, 0] += 15.0 * np.sin(np.linspace(0.0, np.pi, 41))
        result = stabilizer.robust_path_solution(
            anchors,
            np.zeros((n, 2)),
            np.ones(n),
            np.ones(n, bool),
            np.full(n, 0.05),
            np.zeros(n, bool),
            np.zeros(n),
            scale=1.0,
        )

        raw_error = np.max(np.abs(result["raw_solved_center"][:, 0] - 100.0))
        repaired_error = np.max(np.abs(result["center"][:, 0] - 100.0))
        self.assertGreater(raw_error, 10.0)
        self.assertLess(repaired_error, 0.3)
        self.assertTrue(result["excursion_candidate"].any())

    def test_subthreshold_drift_closes_between_absolute_anchors(self):
        n = 80
        anchors = np.zeros((n, 2), dtype=float)
        anchors[:, 0] = np.r_[np.linspace(0.0, 8.0, 40), np.linspace(8.0, 0.0, 40)]
        limb = np.zeros(n, bool)
        limb[[0, -1]] = True
        result = stabilizer.robust_path_solution(
            anchors,
            np.zeros((n, 2)),
            np.ones(n),
            limb,
            np.ones(n),
            ~limb,
            np.full(n, 0.6),
            scale=1.0,
        )

        self.assertGreater(np.max(np.abs(result["raw_solved_center"][:, 0])), 6.0)
        self.assertLess(np.max(np.abs(result["center"][:, 0])), 0.1)
        self.assertAlmostEqual(result["center"][0, 0], 0.0, delta=0.01)
        self.assertAlmostEqual(result["center"][-1, 0], 0.0, delta=0.01)

    def test_confirmed_camera_jump_remains_in_source_path(self):
        n = 60
        source_path = np.tile([100.0, 180.0], (n, 1))
        source_path[30:] += [20.0, -10.0]
        relative = np.zeros((n, 2))
        relative[1:] = np.diff(source_path, axis=0)
        result = stabilizer.robust_path_solution(
            source_path,
            relative,
            np.ones(n),
            np.ones(n, bool),
            np.full(n, 0.5),
            np.zeros(n, bool),
            np.zeros(n),
            scale=1.0,
        )

        np.testing.assert_allclose(result["center"], source_path, atol=1e-5)
        self.assertTrue(result["jump_confirmed"][30])
        target = np.array([135.0, 240.0])
        stabilized_centers = source_path + (target - result["center"])
        np.testing.assert_allclose(stabilized_centers, np.tile(target, (n, 1)), atol=1e-5)

    def test_measurement_jitter_decreases_on_known_camera_path(self):
        n = 100
        x = np.arange(n, dtype=float)
        camera = np.column_stack((100.0 + 0.2 * x, 180.0 + 0.1 * x))
        measured = camera + np.column_stack(
            (1.5 * (-1.0) ** x, 0.8 * (-1.0) ** x)
        )
        relative = np.zeros((n, 2))
        relative[1:] = np.diff(camera, axis=0)
        result = stabilizer.robust_path_solution(
            measured,
            relative,
            np.ones(n),
            np.ones(n, bool),
            np.full(n, 0.5),
            np.zeros(n, bool),
            np.zeros(n),
            scale=1.0,
        )

        before = np.percentile(np.linalg.norm(measured - camera, axis=1), 95)
        after = np.percentile(np.linalg.norm(result["center"] - camera, axis=1), 95)
        self.assertLess(after, 0.65 * before)

    def test_isolated_strong_limb_anchor_outranks_disagreeing_contrast_neighbors(self):
        # A high-quality outer-limb anchor sits alone between two disagreeing
        # contrast-only backup anchors.  Phase motion makes the backup anchors
        # agree with each other, which would otherwise label the limb anchor a
        # transient spike and interpolate it toward their corrupted position.
        n = 100
        i = 50
        anchors = np.full((n, 2), np.nan)
        limb = np.zeros(n, bool)
        contrast = np.zeros(n, bool)
        quality = np.zeros(n)
        score = np.zeros(n)
        limb[i] = True
        quality[i] = 0.8
        anchors[i] = [100.0, 180.0]
        contrast[i - 1] = True
        contrast[i + 1] = True
        score[i - 1] = 0.6
        score[i + 1] = 0.6
        anchors[i - 1] = [121.0, 180.0]
        anchors[i + 1] = [121.5, 180.2]
        relative = np.zeros((n, 2))
        relative[i] = [0.5, 0.2]
        relative[i + 1] = [0.5, 0.2]

        result = stabilizer.robust_path_solution(
            anchors,
            relative,
            np.ones(n),
            limb,
            quality,
            contrast,
            score,
            scale=1.0,
        )

        self.assertFalse(result["jitter_candidate"][i])
        self.assertFalse(result["auto_repaired"][i])
        self.assertLess(np.linalg.norm(result["raw_solved_center"][i] - [100.0, 180.0]), 1.0)
        self.assertLess(
            np.linalg.norm(result["center"][i] - result["raw_solved_center"][i]), 1.0
        )
        self.assertLess(np.linalg.norm(result["center"][i] - [100.0, 180.0]), 1.0)

    def test_long_or_contradictory_gap_is_unresolved(self):
        anchor = np.zeros(220, bool)
        anchor[0] = True
        anchor[-1] = True
        states, _ = stabilizer.classify_centering(anchor, source_fps=30.0)
        self.assertTrue((states[31:-31] == stabilizer.CENTER_UNRESOLVED).all())

        support = np.ones(20, bool)
        conflict = np.zeros(20, bool)
        conflict[8:12] = True
        states, reasons = stabilizer.classify_centering(
            np.zeros(20, bool),
            source_fps=30.0,
            reconstructed_support=support,
            contradictory=conflict,
        )
        self.assertTrue((states[8:12] == stabilizer.CENTER_UNRESOLVED).all())
        self.assertIn("contradictorias", reasons[9])

    def test_no_auto_repair_keeps_raw_solution(self):
        n = 100
        analysis = synthetic_analysis(n)
        analysis["quality"][:] = 0.05
        analysis["raw_center"][30:71, 0] += 15.0 * np.sin(
            np.linspace(0.0, np.pi, 41)
        )
        repaired = stabilizer.solve_tracking(analysis, 0.18, auto_repair=True)
        raw = stabilizer.solve_tracking(analysis, 0.18, auto_repair=False)

        np.testing.assert_allclose(raw["center"], raw["raw_solved_center"])
        self.assertFalse(raw["auto_repaired"].any())
        self.assertGreater(
            np.max(np.linalg.norm(raw["center"] - repaired["center"], axis=1)),
            5.0,
        )
        args = stabilizer.build_parser().parse_args(["--no-auto-repair", "analyze", "video.mp4"])
        self.assertTrue(args.no_auto_repair)

    def test_validate_preview_reports_without_modifying_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview = Path(tmp) / "preview.avi"
            writer = cv2.VideoWriter(
                str(preview), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (120, 160)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                frame = np.zeros((160, 120, 3), np.uint8)
                cv2.circle(frame, (60, 80), 25, (180, 180, 180), -1, cv2.LINE_AA)
                writer.write(frame)
            writer.release()
            analysis = {
                "analysis_width": np.array([120]),
                "analysis_height": np.array([160]),
                "radius": np.array([25.0]),
                "keep": np.ones(8, bool),
                "horizon_start": np.array([8]),
                "center": np.tile([60.0, 80.0], (8, 1)),
            }
            before = analysis["center"].copy()

            result = stabilizer.validate_preview(preview, analysis)

            self.assertGreater(result["validated"], 0)
            np.testing.assert_array_equal(analysis["center"], before)


class GeneralizationPhaseThreeTests(unittest.TestCase):
    def test_parse_frame_spec_normalizes_individuals_ranges_spaces_duplicates(self):
        result = stabilizer.parse_frame_spec("20-24,700, 900,20", 2400)
        expected = np.array(list(range(20, 25)) + [700, 900], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

        np.testing.assert_array_equal(stabilizer.parse_frame_spec("3", 10), [3])
        np.testing.assert_array_equal(stabilizer.parse_frame_spec("1,1,1-2, 2", 10), [1, 2])
        self.assertEqual(len(stabilizer.parse_frame_spec(None, 10)), 0)
        self.assertEqual(len(stabilizer.parse_frame_spec("", 10)), 0)

    def test_parse_frame_spec_rejects_invalid_and_out_of_range(self):
        for bad in ("abc", "1-", "-3", "3-2", "1-5-9", "6000", "99999", "1-6000", "1,,2"):
            with self.assertRaises(SystemExit):
                stabilizer.parse_frame_spec(bad, 2400)

    def test_keep_drop_policy_and_unresolved_rejection(self):
        n = 20
        keep = np.ones(n, bool)
        content = np.full(n, stabilizer.CONTENT_USABLE, np.int8)
        centering = np.full(n, stabilizer.CENTER_RELIABLE, np.int8)
        content[3] = stabilizer.CONTENT_CLIPPED
        keep[3] = False
        centering[15] = stabilizer.CENTER_UNRESOLVED
        analysis = {"keep": keep, "content_state": content, "centering_state": centering}
        original = keep.copy()

        sel = stabilizer.effective_selection(analysis, keep="3")
        self.assertTrue(sel["mask"][3])
        self.assertEqual(sel["origin"][3], "manual")
        self.assertEqual(sel["reason"][3], "recuperado manualmente con --keep-frames")
        # It must never mutate the automatic keep.
        np.testing.assert_array_equal(analysis["keep"], original)

        drop_sel = stabilizer.effective_selection(analysis, drop="3,5")
        self.assertFalse(drop_sel["mask"][3])
        self.assertFalse(drop_sel["mask"][5])
        self.assertEqual(drop_sel["origin"][3], "manual")

        # Conflict between drop and keep is an error, not a silent choice.
        with self.assertRaises(SystemExit):
            stabilizer.effective_selection(analysis, drop="3", keep="3")

        # A keep cannot force an unresolved frame.
        with self.assertRaises(SystemExit):
            stabilizer.effective_selection(analysis, keep="15")
        with self.assertRaises(SystemExit):
            stabilizer.effective_selection(analysis, keep="4")

    def test_contradicted_reconstruction_unresolved_but_backed_movement_kept(self):
        # Regression: reconstructed centering contradicted by UNCERTAIN content,
        # no trusted geometry, a jitter/excursion tracker-error signal, and no
        # confirmed jump must be flagged (and so demoted to CENTER_UNRESOLVED =>
        # never kept).  Genuine geometry-confirmed jumps and directly-measured
        # frames must NOT be flagged.
        def arr(v):
            return np.asarray(v, bool)

        ones = arr(np.ones(1, bool))
        zeros = arr(np.zeros(1, bool))

        # Contradicted: uncertain + reconstructed + no geometry + jitter + !jump.
        flagged = stabilizer.contradicted_reconstruction(
            np.array([stabilizer.CONTENT_UNCERTAIN], np.int8),
            np.array([stabilizer.CENTER_RECONSTRUCTED], np.int8),
            zeros, ones, zeros, zeros,
        )
        self.assertTrue(flagged[0])
        # Also contradicted via the excursion signal, still not a confirmed jump.
        flagged_exc = stabilizer.contradicted_reconstruction(
            np.array([stabilizer.CONTENT_UNCERTAIN], np.int8),
            np.array([stabilizer.CENTER_RECONSTRUCTED], np.int8),
            zeros, zeros, ones, zeros,
        )
        self.assertTrue(flagged_exc[0])

        # Genuine geometry-confirmed jump: must NOT be flagged.
        not_flagged_jump = stabilizer.contradicted_reconstruction(
            np.array([stabilizer.CONTENT_USABLE], np.int8),
            np.array([stabilizer.CENTER_RECONSTRUCTED], np.int8),
            ones, zeros, zeros, ones,
        )
        self.assertFalse(not_flagged_jump[0])
        # Directly-measured uncertain frame (geometry trusted): must NOT be flagged.
        not_flagged_measured = stabilizer.contradicted_reconstruction(
            np.array([stabilizer.CONTENT_UNCERTAIN], np.int8),
            np.array([stabilizer.CENTER_RECONSTRUCTED], np.int8),
            ones, ones, zeros, zeros,
        )
        self.assertFalse(not_flagged_measured[0])

        # End-to-end: the flagged frame is demoted to CENTER_UNRESOLVED so the
        # keep policy drops it, while a backed frame stays exportable.
        keep = np.ones(2, bool)
        centering = np.array([stabilizer.CENTER_RECONSTRUCTED, stabilizer.CENTER_RECONSTRUCTED], np.int8)
        flagged_all = stabilizer.contradicted_reconstruction(
            np.array([stabilizer.CONTENT_UNCERTAIN, stabilizer.CONTENT_USABLE], np.int8),
            centering,
            np.array([False, True]),
            np.array([True, False]),
            np.array([False, False]),
            np.array([False, True]),
        )
        self.assertTrue(flagged_all[0])
        self.assertFalse(flagged_all[1])
        centering[flagged_all] = stabilizer.CENTER_UNRESOLVED
        self.assertEqual(int(centering[0]), stabilizer.CENTER_UNRESOLVED)
        self.assertEqual(int(centering[1]), stabilizer.CENTER_RECONSTRUCTED)
        self.assertFalse(keep[0] and (centering[0] != stabilizer.CENTER_UNRESOLVED))
        self.assertTrue(keep[1] and (centering[1] != stabilizer.CENTER_UNRESOLVED))

    def test_cli_accepts_drop_keep_and_debug_options(self):
        preview = stabilizer.build_parser().parse_args(
            ["preview", "video.mp4", "--drop-frames", "20-24", "--keep-frames", "220"]
        )
        self.assertEqual(preview.video, "video.mp4")
        self.assertEqual(preview.drop_frames, "20-24")
        self.assertEqual(preview.keep_frames, "220")

        export = stabilizer.build_parser().parse_args(
            ["export", "video.mp4", "--drop-frames", "5", "--keep-frames", "3"]
        )
        self.assertEqual(export.drop_frames, "5")
        self.assertEqual(export.keep_frames, "3")

        analyze = stabilizer.build_parser().parse_args(
            ["analyze", "video.mp4", "--debug", "--debug-width", "320", "--debug-max-images", "500"]
        )
        self.assertTrue(analyze.debug)
        self.assertEqual(analyze.debug_width, 320)
        self.assertEqual(analyze.debug_max_images, 500)

    def test_preview_and_export_share_the_same_effective_mask(self):
        analysis = synthetic_analysis(50)
        analysis["clip_edges"][7] = stabilizer.EDGE_BITS["left"]
        analysis["clip_edges"][3] = stabilizer.EDGE_BITS["left"]
        analysis["clip_edges"][10] = stabilizer.EDGE_BITS["right"]
        analysis["clip_score"][7] = 0.9
        solved = stabilizer.solve_tracking(analysis, 0.18)
        preview_args = stabilizer.build_parser().parse_args(
            ["preview", "video.mp4", "--drop-frames", "7", "--keep-frames", "3,10"]
        )
        export_args = stabilizer.build_parser().parse_args(
            ["export", "video.mp4", "--drop-frames", "7", "--keep-frames", "3,10"]
        )
        captured = {}

        def fake_export(*args, **kwargs):
            captured["mask"] = kwargs["selection"]["mask"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir(parents=True, exist_ok=True)
            video = Path(tmp) / "v.mp4"
            info = stabilizer.VideoInfo(240, 360, 50, 30.0, 50 / 30.0)
            orig_load = stabilizer.load_or_analyze
            orig_export = stabilizer.export_video
            try:
                stabilizer.load_or_analyze = lambda a, v, i, o: solved
                stabilizer.export_video = fake_export
                stabilizer.command_preview(preview_args, video, info, out)
                mask1 = captured["mask"]
                stabilizer.command_export(export_args, video, info, out)
                mask2 = captured["mask"]
            finally:
                stabilizer.load_or_analyze = orig_load
                stabilizer.export_video = orig_export
        np.testing.assert_array_equal(mask1, mask2)
        self.assertFalse(mask1[7])
        self.assertTrue(mask1[3])
        self.assertTrue(mask1[10])

    def _write_synthetic_video(self, tmp, n, width=240, height=360, fps=30.0):
        path = Path(tmp) / "synth.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            self.skipTest("no se pudo crear el vídeo sintético")
        for _ in range(n):
            img = np.zeros((height, width, 3), np.uint8)
            cv2.circle(img, (width // 2, height // 2), 28, (200, 200, 200), -1, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        return path

    def _debug_analysis(self, n):
        analysis = synthetic_analysis(n)
        analysis["clip_edges"][2] = stabilizer.EDGE_BITS["left"]
        analysis["clip_edges"][3] = stabilizer.EDGE_BITS["right"]
        analysis["clip_score"][2] = 0.9
        analysis["clip_score"][3] = 0.9
        analysis["radial_points"][5] = 5
        analysis["radial_points"][6] = 5
        analysis["quality"][5] = 0.01
        analysis["quality"][6] = 0.01
        analysis.update(stabilizer.solve_tracking(analysis, 0.18))
        return analysis

    def test_debug_produces_complete_review_csv_and_all_jpegs_under_limit(self):
        n = 40
        with tempfile.TemporaryDirectory() as tmp:
            video = self._write_synthetic_video(tmp, n)
            info = stabilizer.VideoInfo(240, 360, n, 30.0, n / 30.0)
            analysis = self._debug_analysis(n)
            out = Path(tmp) / "out"
            stabilizer.write_debug(video, info, analysis, out, debug_width=320, max_images=10000)

            debug_dir = out / "debug"
            self.assertTrue((debug_dir / "review.mp4").exists())
            self.assertTrue((debug_dir / "frames.csv").exists())
            self.assertTrue((debug_dir / "contact_sheet.jpg").exists())

            cap = cv2.VideoCapture(str(debug_dir / "review.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n)
            cap.release()

            with (debug_dir / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), n + 1)

            jpgs = list((debug_dir / "frames").rglob("*.jpg"))
            self.assertLessEqual(len(jpgs), 10000)
            self.assertEqual(len(jpgs), n)
            self.assertTrue(any("clipped" in p.name for p in jpgs))
            self.assertTrue(any("uncertain" in p.name for p in jpgs))
            self.assertTrue(any("usable" in p.name for p in jpgs))

    def test_debug_small_limit_prioritizes_and_stays_under_limit(self):
        n = 30
        limit = 6
        with tempfile.TemporaryDirectory() as tmp:
            video = self._write_synthetic_video(tmp, n)
            info = stabilizer.VideoInfo(240, 360, n, 30.0, n / 30.0)
            analysis = self._debug_analysis(n)
            out = Path(tmp) / "out"
            stabilizer.write_debug(video, info, analysis, out, debug_width=320, max_images=limit)

            debug_dir = out / "debug"
            jpgs = list((debug_dir / "frames").rglob("*.jpg"))
            self.assertLessEqual(len(jpgs), limit)
            names = [p.name for p in jpgs]
            for expected in (
                "frame_00002_clipped.jpg",
                "frame_00003_clipped.jpg",
                "frame_00005_uncertain.jpg",
                "frame_00006_uncertain.jpg",
            ):
                self.assertIn(expected, names)

            # review and CSV stay complete even when the JPEGs are sampled.
            cap = cv2.VideoCapture(str(debug_dir / "review.mp4"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n)
            cap.release()
            with (debug_dir / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), n + 1)


class GeneralizationPhaseFourTests(unittest.TestCase):
    def test_cli_requires_positional_video_in_every_subcommand(self):
        for command in ("inspect", "analyze", "preview", "export"):
            with self.assertRaises(SystemExit):
                stabilizer.build_parser().parse_args([command])
            args = stabilizer.build_parser().parse_args([command, "video.mp4"])
            self.assertEqual(args.video, "video.mp4")

    def test_default_output_dir_is_distinct_and_safe_per_stem(self):
        out1 = stabilizer.default_output_dir(stabilizer.ROOT / "foo bar.MP4")
        out2 = stabilizer.default_output_dir(stabilizer.ROOT / "other.mp4")
        collision = stabilizer.default_output_dir(stabilizer.ROOT / "foo_bar.mp4")
        self.assertNotEqual(out1, out2)
        self.assertNotEqual(out1, collision)
        self.assertRegex(out1.name, r"^foo_bar_[0-9a-f]{8}_output$")
        self.assertTrue(str(out1).startswith(str(stabilizer.ROOT)))

    def _write_identity_cache(self, path, identity):
        arrays = {key: np.zeros(1) for key in stabilizer.CACHE_REQUIRED_ARRAYS}
        arrays.update(stabilizer.identity_arrays(identity))
        np.savez_compressed(path, **arrays)

    def test_cache_identity_rejects_different_video_width_radius_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            v2 = tmp / "b.mp4"
            v2.write_bytes(b"y" * 2048)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            base = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            path = tmp / "analysis.npz"
            self._write_identity_cache(path, base)
            self.assertEqual(stabilizer.cache_status(path, base)[0], "valid")
            other_video = stabilizer.build_cache_identity(v2, info, 270, 480, "auto", None, None, 0.18)
            self.assertEqual(stabilizer.cache_status(path, other_video)[0], "incompatible")
            changed = stabilizer.build_cache_identity(v1, info, 320, 568, "auto", None, None, 0.18)
            self.assertEqual(stabilizer.cache_status(path, changed)[0], "incompatible")
            changed = stabilizer.build_cache_identity(v1, info, 270, 480, "45.5", None, None, 0.18)
            self.assertEqual(stabilizer.cache_status(path, changed)[0], "incompatible")
            profile = {
                "version": 1,
                "discards": [(1, 2)],
            }
            changed = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", profile, "prof.json", 0.18)
            self.assertEqual(stabilizer.cache_status(path, changed)[0], "incompatible")
            self.assertEqual(stabilizer.cache_status(path, base)[0], "valid")

    def test_cache_identity_profile_path_and_hash_participate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            profile = {
                "version": 1,
                "discards": [(1, 2)],
            }
            p1 = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", profile, "a.json", 0.18)
            p2 = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", profile, "b.json", 0.18)
            p3 = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            self.assertNotEqual(p1, p2)  # different profile path
            self.assertNotEqual(p1["profile_hash"], p3["profile_hash"])  # content participates
            self.assertNotEqual(p1["profile_path"], "")

    def test_no_auto_repair_does_not_change_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            identity = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            path = tmp / "analysis.npz"
            self._write_identity_cache(path, identity)
            self.assertEqual(stabilizer.cache_status(path, identity)[0], "valid")
            args = stabilizer.build_parser().parse_args(["--no-auto-repair", "analyze", "video.mp4"])
            self.assertTrue(args.no_auto_repair)

    def test_legacy_cache_without_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            path = tmp / "analysis.npz"
            np.savez_compressed(path, raw_center=np.zeros((10, 2)), quality=np.zeros(10))
            expected = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            status, reasons = stabilizer.cache_status(path, expected)
            self.assertEqual(status, "incompatible")
            self.assertTrue(any("legacy" in reason for reason in reasons))

    def test_cache_status_distinguishes_contrast_only_refreshable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            expected = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            path = tmp / "analysis.npz"
            complete = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            stale = dict(expected)
            stale["contrast_version"] = int(expected["contrast_version"]) - 1
            complete.update(stabilizer.identity_arrays(stale))
            np.savez_compressed(path, **complete)
            status, reasons = stabilizer.cache_status(path, expected)
            self.assertEqual(status, "refreshable")
            self.assertTrue(any("contrast_version" in reason for reason in reasons))
            # A hard non-contrast identity mismatch stays incompatible.
            changed = dict(expected)
            changed["width"] = 400
            complete2 = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            complete2.update(stabilizer.identity_arrays(changed))
            np.savez_compressed(path, **complete2)
            status, reasons = stabilizer.cache_status(path, expected)
            self.assertEqual(status, "incompatible")
            self.assertTrue(any("'width' difiere" in reason for reason in reasons))

    def test_cache_status_reports_missing_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            expected = stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)
            path = tmp / "analysis.npz"
            self.assertEqual(stabilizer.cache_status(path, expected)[0], "missing")
            np.savez_compressed(path, **stabilizer.identity_arrays(expected))
            status, reasons = stabilizer.cache_status(path, expected)
            self.assertEqual(status, "incompatible")
            self.assertTrue(any("incompleta" in reason for reason in reasons))

            complete = {key: np.zeros(1) for key in stabilizer.CACHE_REQUIRED_ARRAYS}
            complete.update(stabilizer.identity_arrays(expected))
            np.savez_compressed(path, **complete)
            self.assertEqual(stabilizer.cache_status(path, expected)[0], "valid")

    def test_export_preserves_source_width_height_and_fps(self):
        n = 24
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, n, width=240, height=360, fps=30.0)
            info = stabilizer.probe_video(video)
            keep = np.ones(n, bool)
            keep[3] = False
            analysis = {
                "analysis_width": np.array([info.width]),
                "analysis_height": np.array([info.height]),
                "trim_end": np.array([n - 1]),
                "keep": keep,
                "center": np.tile([info.width / 2.0, info.height / 2.0], (n, 1)),
            }
            out = Path(tmp) / "out"
            dest = out / "stabilized.mp4"
            stabilizer.export_video(
                video, info, analysis, dest, info.width, info.height,
                speed=1.0, crf=20, preset="ultrafast", debug_overlay=False,
            )
            probe = stabilizer.probe_video(dest)
            self.assertEqual(probe.width, info.width)
            self.assertEqual(probe.height, info.height)
            self.assertEqual(probe.frames, n - 1)
            self.assertAlmostEqual(probe.fps, info.fps, delta=0.6)

    def test_analysis_summary_reports_all_state_counts_and_profile_discards(self):
        analysis = synthetic_analysis(40)
        analysis["clip_edges"][2] = stabilizer.EDGE_BITS["left"]
        analysis["clip_score"][2] = 0.9
        analysis["radial_points"][5] = 5
        analysis["quality"][5] = 0.01
        analysis["raw_center"][20:] = np.nan
        analysis["radial_points"][20:] = 0
        analysis["contrast_center"][20:] = np.nan
        analysis["contrast_score"][20:] = 0.0
        analysis["maximum"][20:] = 200
        profile = {
            "version": 1,
            "discards": [(30, 33)],
        }
        solved = stabilizer.solve_tracking(analysis, 0.18, profile)
        full = analysis.copy()
        full.update(solved)
        info = stabilizer.VideoInfo(240, 360, 40, 30.0, 40 / 30.0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stabilizer.print_analysis_summary(full, info)
        text = buffer.getvalue()
        self.assertIn("contenido: 18 usable, 1 recortado, 21 incierto", text)
        self.assertIn("centrado: 16 fiable, 24 reconstruido, 0 sin resolver", text)
        self.assertIn("frames escritos/descartados: 35/5", text)
        self.assertIn("descartados por perfil: 4", text)

    def test_analysis_summary_reports_manual_decisions_when_selection_given(self):
        analysis = synthetic_analysis(30)
        analysis["clip_edges"][3] = stabilizer.EDGE_BITS["left"]
        analysis["clip_score"][3] = 0.9
        solved = stabilizer.solve_tracking(analysis, 0.18)
        full = analysis.copy()
        full.update(solved)
        selection = stabilizer.effective_selection(full, drop="5", keep="3")
        info = stabilizer.VideoInfo(240, 360, 30, 30.0, 30 / 30.0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stabilizer.print_analysis_summary(full, info, selection=selection)
        text = buffer.getvalue()
        self.assertIn("frames escritos/descartados: 29/1", text)
        self.assertIn("decisiones manuales: 1 recuperados, 1 descartados", text)


class PhaseTwoArcTrackingTests(unittest.TestCase):
    """Consolidated Phase 2 tracker coverage: bootstrap + phase-predicted
    translation, large confirmed camera jump with contour confirmation, and
    the no-measurement semantics (contour without radial evidence, blank
    frames, unrelated touching contour)."""

    def _crescent(self, center, radius=68, size=(360, 240), angle_deg=60, peak=175):
        image = np.zeros(size, np.uint8)
        c = tuple(np.rint(center).astype(int))
        cv2.circle(image, c, int(radius), peak, -1, cv2.LINE_AA)
        angle = np.deg2rad(angle_deg)
        moon_center = (
            int(round(center[0] + 0.72 * radius * np.cos(angle))),
            int(round(center[1] + 0.72 * radius * np.sin(angle))),
        )
        cv2.circle(image, moon_center, int(radius), 0, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(image, (3, 3), 0.6)

    def test_bootstrap_and_phase_predicted_translation(self):
        positions = [
            np.array([90.0, 151.0]),
            np.array([104.0, 160.0]),
            np.array([118.0, 169.0]),
            np.array([132.0, 178.0]),
        ]
        tracker = stabilizer.ArcGeometryTracker(radius=68.0, scale=1.0)
        first = tracker.step(
            self._crescent(positions[0], angle_deg=40), np.array([0.0, 0.0]), 0.0, positions[0]
        )
        self.assertTrue(first["accepted"])
        self.assertIsNotNone(tracker.last_accepted)
        for i in range(1, len(positions)):
            delta = positions[i] - positions[i - 1]
            # contour=None: only the primary radial measurement around the
            # phase-predicted location can supply the center.
            result = tracker.step(self._crescent(positions[i], angle_deg=40 + i * 15), delta, 1.0, None)
            self.assertTrue(result["accepted"], f"frame {i}")
            self.assertEqual(result["source"], stabilizer.SRC_RADIAL, f"frame {i}")
            self.assertLess(np.linalg.norm(result["center"] - positions[i]), 1.5, f"frame {i}")

    def test_large_camera_jump_and_contour_confirmation(self):
        c0 = np.array([121.0, 180.0])
        c1 = np.array([201.0, 180.0])  # +80 px real camera jump
        tracker = stabilizer.ArcGeometryTracker(radius=68.0, scale=1.0)
        tracker.step(self._crescent(c0), np.array([0.0, 0.0]), 0.0, c0)
        # Phase reports the real jump; the contour candidate sits at the new
        # location and has radial evidence there.
        result = tracker.step(self._crescent(c1), np.array([80.0, 0.0]), 0.9, c1)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["source"], stabilizer.SRC_CONTOUR)
        self.assertLess(np.linalg.norm(result["center"] - c1), 1.5)
        # The selected arc evidence is the confirming radial measurement at the
        # contour center, never the failed attempt around the old prediction.
        self.assertTrue(result["arc"]["measured"])
        self.assertLess(np.linalg.norm(result["arc"]["center"] - c1), 1.5)
        self.assertFalse(result["attempted_arc"]["measured"])
        # Jump not clamped: tracker re-seeded at the new location.
        self.assertLess(np.linalg.norm(tracker.last_accepted - c1), 1.5)

    def test_no_measurement_without_radial_evidence(self):
        c0 = np.array([121.0, 180.0])
        # A frame with no radial evidence (blank) never returns a measurement
        # and never mutates accepted state; propagated prediction is not one.
        tracker = stabilizer.ArcGeometryTracker(radius=68.0, scale=1.0)
        ok = tracker.step(self._crescent(c0), np.array([0.0, 0.0]), 0.0, c0)
        self.assertTrue(ok["accepted"])
        self.assertEqual(ok["source"], stabilizer.SRC_RADIAL)
        blank = np.zeros((360, 240), np.uint8)
        accepted_center = tracker.last_accepted.copy()
        gone = tracker.step(blank, np.array([0.0, 0.0]), 1.0, None)
        self.assertFalse(gone["accepted"])
        self.assertEqual(gone["source"], stabilizer.SRC_NONE)
        self.assertIsNone(gone["center"])
        # The blank frame does not move the accepted center.
        np.testing.assert_allclose(tracker.last_accepted, accepted_center, atol=1e-9)
        # A wrong contour with no phase motion and no radial evidence is rejected.
        tracker2 = stabilizer.ArcGeometryTracker(radius=68.0, scale=1.0)
        tracker2.step(self._crescent(c0), np.array([0.0, 0.0]), 0.0, c0)
        res = tracker2.step(
            self._crescent(c0 + np.array([80.0, 0.0])), np.array([0.0, 0.0]), 1.0, c0 + np.array([3.0, 0.0])
        )
        self.assertFalse(res["accepted"])
        self.assertEqual(res["source"], stabilizer.SRC_NONE)
        self.assertLess(np.linalg.norm(tracker2.last_accepted - c0), 2.0)
        # An unrelated touching contour must not invalidate a selected radial.
        tracker3 = stabilizer.ArcGeometryTracker(radius=68.0, scale=1.0)
        tracker3.step(self._crescent(c0), np.array([0.0, 0.0]), 0.0, None)
        touch = tracker3.step(
            self._crescent(c0), np.array([0.0, 0.0]), 1.0, c0 + np.array([2.0, 2.0]), contour_touch=True
        )
        self.assertEqual(touch["source"], stabilizer.SRC_RADIAL)
        self.assertTrue(touch["accepted"])


class ContinuousArcCacheTests(unittest.TestCase):
    """Consolidated cache-version/required-array and invalidation coverage for
    the continuous-arc detections and the anchored contrast fallback."""

    def _expected(self, tmp):
        v1 = Path(tmp) / "a.mp4"
        v1.write_bytes(b"x" * 1024)
        info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
        return stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)

    def test_cache_versions_and_required_arrays(self):
        self.assertEqual(stabilizer.CACHE_VERSION, 6)
        self.assertEqual(stabilizer.TEAR_VERSION, 1)
        self.assertGreaterEqual(stabilizer.CONTRAST_VERSION, 3)
        for name in (
            "arc_center", "arc_measured", "arc_valid_points", "arc_coverage",
            "arc_median_residual", "arc_strength", "arc_gap_deg", "arc_gap_angle",
            "geometry_source", "geometry_prediction", "geometry_innovation",
            "contrast_dynamic_offset", "contrast_offset_sample", "fallback_reanchored",
            "fallback_supported", "fallback_innovation", "fallback_mode",
            "tear_evaluable", "tear_bright_level", "tear_visible_threshold",
            "tear_exposed_fraction", "tear_reason",
        ):
            self.assertIn(name, stabilizer.CACHE_REQUIRED_ARRAYS, name)
        self.assertNotIn("timed_repaired", stabilizer.CACHE_REQUIRED_ARRAYS)

    def test_cache_invalidates_when_arrays_missing_or_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._expected(tmp)
            path = Path(tmp) / "analysis.npz"
            # Missing a required arc array -> rejected as incomplete.
            incomplete = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            del incomplete["geometry_source"]
            incomplete.update(stabilizer.identity_arrays(expected))
            np.savez_compressed(path, **incomplete)
            status, reasons = stabilizer.cache_status(path, expected)
            self.assertEqual(status, "incompatible")
            self.assertTrue(any("incompleta" in reason for reason in reasons))
            # Legacy cache without the new fallback arrays -> rejected.
            legacy = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            del legacy["contrast_dynamic_offset"]
            del legacy["fallback_reanchored"]
            legacy.update(stabilizer.identity_arrays(expected))
            np.savez_compressed(path, **legacy)
            self.assertEqual(stabilizer.cache_status(path, expected)[0], "incompatible")
            # Complete cache -> valid.
            complete = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            complete.update(stabilizer.identity_arrays(expected))
            np.savez_compressed(path, **complete)
            self.assertEqual(stabilizer.cache_status(path, expected)[0], "valid")

    def test_analyze_video_refreshable_contrast_only_cache_keeps_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)
            info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0)
            args = stabilizer.build_parser().parse_args(["analyze", "video.mp4"])
            analysis_w, analysis_h = stabilizer.scaled_shape(info, args.analysis_width)
            expected = stabilizer.build_cache_identity(
                v1, info, analysis_w, analysis_h, "auto", None, None, args.min_quality
            )
            complete = {k: np.zeros(1) for k in stabilizer.CACHE_REQUIRED_ARRAYS}
            stale = dict(expected)
            stale["contrast_version"] = int(expected["contrast_version"]) - 1
            complete.update(stabilizer.identity_arrays(stale))
            path = tmp / "analysis.npz"
            np.savez_compressed(path, **complete)
            captured = io.StringIO()
            with mock.patch("eclipse_stabilizer_core.analysis.refresh_contrast_track") as refresh:
                with mock.patch("eclipse_stabilizer_core.analysis.solve_tracking", return_value={}):
                    with mock.patch("eclipse_stabilizer_core.analysis.save_analysis_cache"):
                        with mock.patch("eclipse_stabilizer_core.analysis.write_tracking_csv"):
                            with mock.patch(
                                "eclipse_stabilizer_core.analysis._geometry_trusted_mask",
                                return_value=np.ones(100, bool),
                            ):
                                with mock.patch("eclipse_stabilizer_core.analysis.resolve_radius") as resolve:
                                    with contextlib.redirect_stdout(captured):
                                        analysis = stabilizer.analyze_video(args, v1, info, tmp)
            self.assertEqual(refresh.call_count, 1)
            self.assertFalse(resolve.called)
            self.assertEqual(analysis["raw_center"].shape, (1,))
            text = captured.getvalue()
            self.assertIn("Cargando análisis cacheado", text)
            self.assertIn("recalculando el respaldo de contraste", text)
            self.assertNotIn("Caché rechazada", text)
            self.assertNotIn("Regenerando análisis completo", text)


class PhaseThreeAnchoredFallbackTests(unittest.TestCase):
    """Consolidated Phase 3 anchored-fallback coverage: re-anchor + learned
    evolving offset, short-gap bridging + bounded bad innovation, and real-jump
    preservation + blank/no-history never becoming a reliable measurement."""

    def _disk(self, center, radius=16, size=(360, 240)):
        image = np.zeros(size, np.uint8)
        cv2.circle(image, tuple(np.rint(center).astype(int)), radius, 180, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(image, (3, 3), 0.6)

    def _blank(self, size=(360, 240)):
        return np.zeros(size, np.uint8)

    def _analysis(self, n, centers, sources, relatives, responses, fps=30.0):
        raw = np.full((n, 2), np.nan, np.float64)
        for i, c in enumerate(centers):
            if c is not None:
                raw[i] = c
        return {
            "analysis_width": np.array([270]),
            "analysis_height": np.array([480]),
            "source_frames": np.array([n]),
            "source_fps": np.array([fps]),
            "radius": np.array([68.0]),
            "raw_center": raw,
            "relative": np.asarray(relatives, np.float64).reshape(n, 2),
            "response": np.asarray(responses, np.float64).reshape(n),
            "geometry_source": np.asarray(sources, np.int8),
        }

    def _run(self, analysis, frames):
        return stabilizer.refresh_contrast_track_frames(analysis, frames)

    def test_fallback_reanchors_and_learns_evolving_offset(self):
        # Every reliable geometry frame re-anchors exactly to raw_center.
        centers = [np.array([100.0, 150.0]), np.array([121.0, 180.0]), np.array([140.0, 200.0])]
        analysis = self._analysis(len(centers), centers, [stabilizer.SRC_RADIAL] * len(centers),
                                  [[0.0, 0.0]] * len(centers), [1.0] * len(centers))
        result = self._run(analysis, [self._blank() for _ in centers])
        self.assertTrue(result["fallback_reanchored"].all())
        for i, c in enumerate(centers):
            np.testing.assert_allclose(result["contrast_center"][i], c, atol=1e-9)
        # A slowly evolving crescent offset is learned by the robust history.
        g = np.array([121.0, 180.0])
        n = 20
        offsets = [np.array([4.0, 0.0]) + i * np.array([0.6, 0.3]) for i in range(n)]
        analysis2 = self._analysis(n, [g] * n, [stabilizer.SRC_RADIAL] * n, [[0.0, 0.0]] * n, [1.0] * n)
        result2 = self._run(analysis2, [self._disk(g - o) for o in offsets])
        samples = result2["contrast_offset_sample"]
        recent = result2["contrast_offset"]
        self.assertTrue(np.isfinite(recent).all())
        self.assertLess(np.linalg.norm(recent - np.median(samples, axis=0)), 1.0)
        self.assertGreater(np.linalg.norm(recent - samples[0]), 2.0)

    def test_fallback_bridges_gap_and_bounds_innovation(self):
        # A short SRC_NONE gap is propagated smoothly between geometry anchors.
        c0 = np.array([100.0, 150.0])
        gd = [np.array([3.0, 2.0]), np.array([3.0, 2.0])]
        c_end = c0 + gd[0] + gd[1] + np.array([2.0, 1.0])
        centers = [c0, c0 + gd[0], None, None, c_end]
        sources = [stabilizer.SRC_RADIAL, stabilizer.SRC_RADIAL, stabilizer.SRC_NONE,
                   stabilizer.SRC_NONE, stabilizer.SRC_RADIAL]
        rel = [[0.0, 0.0], gd[0], gd[1], [2.0, 1.0], [2.0, 1.0]]
        analysis = self._analysis(5, centers, sources, rel, [1.0] * 5)
        result = self._run(analysis, [self._blank() for _ in range(5)])
        track = result["contrast_center"]
        np.testing.assert_allclose(track[3] - track[2], rel[3], atol=1e-9)
        self.assertLess(np.linalg.norm(np.diff(track, axis=0), axis=1).max(), 6.0)
        np.testing.assert_allclose(track[4], c_end, atol=1e-9)
        # A bad contrast detection on a propagation frame is robustly capped.
        g = np.array([121.0, 180.0])
        nw = 8
        warm = self._analysis(nw, [g] * nw, [stabilizer.SRC_RADIAL] * nw,
                              [[0.0, 0.0]] * nw, [1.0] * nw)
        warm_frames = [self._disk(g - np.array([8.0, 0.0])) for _ in range(nw)]
        self._run(warm, warm_frames)
        analysis2 = self._analysis(nw + 1, [g] * nw + [None],
                                   [stabilizer.SRC_RADIAL] * nw + [stabilizer.SRC_NONE],
                                   [[0.0, 0.0]] * nw + [[0.0, 0.0]], [1.0] * (nw + 1))
        bad = self._disk(g + np.array([35.0, 0.0]), radius=14)
        result2 = self._run(analysis2, warm_frames + [bad])
        last = result2["contrast_center"][-1]
        self.assertLess(np.linalg.norm(last - g), 4.0)
        self.assertGreater(np.linalg.norm(last - (g + np.array([35.0, 0.0]))), 30.0)
        self.assertLess(result2["fallback_innovation"][-1], 4.0)

    def test_fallback_real_jump_preserved_and_no_history_unsupported(self):
        # A real camera jump confirmed by geometry is preserved (re-anchored).
        c0 = np.array([121.0, 180.0])
        c1 = np.array([201.0, 180.0])
        analysis = self._analysis(2, [c0, c1], [stabilizer.SRC_RADIAL, stabilizer.SRC_RADIAL],
                                  [[0.0, 0.0], [80.0, 0.0]], [1.0, 1.0])
        result = self._run(analysis, [self._blank(), self._blank()])
        np.testing.assert_allclose(result["contrast_center"][1], c1, atol=1e-9)
        # Blank / no-history emits no finite center and stays unsupported.
        n = 5
        blank_an = self._analysis(n, [None] * n, [stabilizer.SRC_NONE] * n,
                                  [[0.0, 0.0]] * n, [1.0] * n)
        blank_res = self._run(blank_an, [self._blank() for _ in range(n)])
        self.assertFalse(blank_res["fallback_reanchored"].any())
        self.assertFalse(blank_res["fallback_supported"].any())
        self.assertTrue(np.isnan(blank_res["contrast_center"]).all())
        np.testing.assert_allclose(blank_res["contrast_offset"], np.zeros(2), atol=1e-12)


class PhaseThreeSolverTests(unittest.TestCase):
    """Consolidated Phase 3 solver coverage: conservative spike closure
    (coherent corruption repaired / contradictory neighbours not), confirmed
    camera jump preserved, and an unsupported blank sequence never exported."""

    def _spike(self, rel20, rel21, n=40, i=20, score=1.0, neighbor_shift=0.0):
        anchors = np.tile([120.0, 180.0], (n, 1))
        if neighbor_shift:
            anchors[i + 1] += [neighbor_shift, 0.0]
        limb = np.ones(n, bool)
        contrast = np.zeros(n, bool)
        sc = np.zeros(n)
        contrast[i] = True
        limb[i] = False
        sc[i] = score
        rel = np.zeros((n, 2))
        rel[i] = rel20
        rel[i + 1] = rel21
        return anchors, rel, limb, sc, contrast

    def test_conservative_spike_closure(self):
        # Marginally coherent isolated corruption is repaired back toward the
        # coherent neighbours (neighbour error within 2 * gate).
        a, rel, limb, sc, contrast = self._spike([150.0, -150.0], [-150.0, 150.0])
        result = stabilizer.robust_path_solution(
            a, rel, np.ones(len(a)), limb, np.full(len(a), 0.8), contrast, sc, scale=1.0
        )
        self.assertTrue(result["jitter_candidate"][20])
        self.assertTrue(result["auto_repaired"][20])
        raw_dist = np.linalg.norm(result["raw_solved_center"][20] - [120.0, 180.0])
        center_dist = np.linalg.norm(result["center"][20] - [120.0, 180.0])
        self.assertLess(center_dist, 0.5 * raw_dist)
        # Contradictory neighbours outside the closure gate are NOT repaired.
        a2, rel2, limb2, sc2, contrast2 = self._spike([150.0, -150.0], [-130.0, 150.0], neighbor_shift=10.0)
        result2 = stabilizer.robust_path_solution(
            a2, rel2, np.ones(len(a2)), limb2, np.full(len(a2), 0.8), contrast2, sc2, scale=1.0
        )
        self.assertFalse(result2["jitter_candidate"][20])
        self.assertFalse(result2["auto_repaired"][20])

    def test_confirmed_camera_jump_is_preserved(self):
        n = 40
        source = np.tile([120.0, 180.0], (n, 1))
        source[20:] += [80.0, 0.0]
        relative = np.zeros((n, 2))
        relative[1:] = np.diff(source, axis=0)
        result = stabilizer.robust_path_solution(
            source, relative, np.ones(n), np.ones(n, bool),
            np.full(n, 0.8), np.zeros(n, bool), np.zeros(n), scale=1.0,
        )
        self.assertTrue(result["jump_confirmed"][20])
        np.testing.assert_allclose(result["center"], source, atol=1e-5)

    def test_unsupported_blank_never_exported(self):
        # A completely blank, unsupported sequence has no reliable anchor.
        analysis = synthetic_analysis(50)
        analysis["raw_center"][:] = np.nan
        analysis["contrast_center"][:] = np.nan
        analysis["contrast_score"][:] = 0.0
        analysis["fallback_supported"] = np.zeros(50, bool)
        with self.assertRaises(SystemExit):
            stabilizer.solve_tracking(analysis, 0.18)
        # An unsupported blank prefix is not treated as contrast backup support.
        analysis2 = synthetic_analysis(50)
        analysis2["raw_center"][:10] = np.nan
        analysis2["contrast_center"][:10] = np.nan
        analysis2["contrast_score"][:10] = 0.0
        analysis2["fallback_supported"] = np.zeros(50, bool)
        analysis2["fallback_supported"][10:] = True
        result2 = stabilizer.solve_tracking(analysis2, 0.18)
        self.assertFalse(result2["horizon_tracked"][:10].any())
        for i in range(10):
            self.assertNotIn("respaldo de contraste", result2["centering_reason"][i])


class ProbeJsonTests(unittest.TestCase):
    """Synthetic ffprobe metadata parsing and validation, no external processes."""

    def _json(self, streams, fmt=None):
        return json.dumps({"streams": streams, "format": fmt or {}})

    def _stream(self, **overrides):
        base = {
            "width": 720,
            "height": 1280,
            "avg_frame_rate": "30/1",
            "r_frame_rate": "30/1",
            "nb_frames": "2400",
            "duration": "80.0",
        }
        base.update(overrides)
        return base

    def test_complete_valid_probe_returns_exact_values(self):
        info, estimated = stabilizer.parse_probe_json(
            self._json([self._stream(avg_frame_rate="2400/80")])
        )
        self.assertFalse(estimated)
        self.assertEqual((info.width, info.height, info.frames), (720, 1280, 2400))
        self.assertAlmostEqual(info.fps, 2400 / 80, places=6)
        self.assertAlmostEqual(info.duration, 80.0, places=6)

    def test_zero_over_zero_fps_falls_back_to_r_frame_rate(self):
        info, estimated = stabilizer.parse_probe_json(
            self._json([self._stream(avg_frame_rate="0/0", r_frame_rate="30000/1001")])
        )
        self.assertAlmostEqual(info.fps, 30000 / 1001, places=6)
        self.assertFalse(estimated)

    def test_missing_avg_frame_rate_uses_r_frame_rate(self):
        stream = self._stream()
        del stream["avg_frame_rate"]
        info, _ = stabilizer.parse_probe_json(self._json([stream]))
        self.assertAlmostEqual(info.fps, 30.0, places=6)

    def test_missing_duration_is_derived_from_frames_and_fps(self):
        stream = self._stream()
        del stream["duration"]
        info, _ = stabilizer.parse_probe_json(self._json([stream]))
        self.assertAlmostEqual(info.duration, 2400 / 30.0, places=6)

    def test_na_nb_frames_is_estimated_and_identified(self):
        info, estimated = stabilizer.parse_probe_json(
            self._json([self._stream(nb_frames="N/A")])
        )
        self.assertTrue(estimated)
        self.assertEqual(info.frames, int(round(80.0 * 30.0)))

    def test_absent_nb_frames_is_estimated_and_identified(self):
        stream = self._stream()
        del stream["nb_frames"]
        info, estimated = stabilizer.parse_probe_json(self._json([stream]))
        self.assertTrue(estimated)
        self.assertEqual(info.frames, int(round(80.0 * 30.0)))

    def test_zero_or_negative_nb_frames_is_rejected(self):
        for bad in ("0", "-3", 0):
            with self.subTest(nb_frames=bad):
                with self.assertRaises(SystemExit):
                    stabilizer.parse_probe_json(self._json([self._stream(nb_frames=bad)]))

    def test_nan_or_inf_nb_frames_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(nb_frames=bad):
                with self.assertRaises(SystemExit):
                    stabilizer.parse_probe_json(self._json([self._stream(nb_frames=bad)]))

    def test_invalid_fps_fractions_fall_back_to_r_frame_rate(self):
        for bad in ("0/0", "nan/1", "inf/1", "-inf/1", "1/0"):
            with self.subTest(avg_frame_rate=bad):
                info, _ = stabilizer.parse_probe_json(
                    self._json([self._stream(avg_frame_rate=bad, r_frame_rate="30000/1001")])
                )
                self.assertAlmostEqual(info.fps, 30000 / 1001, places=6)

    def test_fps_derived_from_frames_and_duration_when_both_fractions_invalid(self):
        stream = {
            "width": 720,
            "height": 1280,
            "avg_frame_rate": "0/0",
            "r_frame_rate": "N/A",
            "nb_frames": "2400",
            "duration": "80.0",
        }
        info, estimated = stabilizer.parse_probe_json(self._json([stream]))
        self.assertAlmostEqual(info.fps, 2400 / 80.0, places=6)
        self.assertFalse(estimated)

    def test_invalid_format_duration_falls_back_to_stream_duration(self):
        stream = self._stream()
        stream["duration"] = "80.0"
        info, _ = stabilizer.parse_probe_json(self._json([stream], fmt={"duration": float("inf")}))
        self.assertAlmostEqual(info.duration, 80.0, places=6)

    def test_nan_inf_durations_are_skipped_and_frames_derive_duration(self):
        stream = self._stream()
        stream["duration"] = float("nan")
        info, _ = stabilizer.parse_probe_json(self._json([stream], fmt={"duration": float("inf")}))
        self.assertAlmostEqual(info.duration, 2400 / 30.0, places=6)

    def test_empty_streams_raises_clear_error(self):
        with self.assertRaises(SystemExit) as ctx:
            stabilizer.parse_probe_json(self._json([]))
        self.assertIn("stream de vídeo", str(ctx.exception))

    def test_invalid_dimensions_raise_clear_error(self):
        for key, bad in (("width", 0), ("width", -5), ("width", "N/A"), ("height", "abc"), ("height", 0.0)):
            with self.subTest(key=key, value=bad):
                with self.assertRaises(SystemExit):
                    stabilizer.parse_probe_json(self._json([self._stream(**{key: bad})]))

    def test_unresolvable_metadata_raises_clear_error(self):
        stream = {
            "width": 720,
            "height": 1280,
            "avg_frame_rate": "0/0",
            "r_frame_rate": "N/A",
            "duration": "N/A",
        }
        with self.assertRaises(SystemExit) as ctx:
            stabilizer.parse_probe_json(self._json([stream]))
        self.assertIn("FPS", str(ctx.exception))

    def test_no_duration_no_frames_raises_clear_error(self):
        stream = {
            "width": 720,
            "height": 1280,
            "avg_frame_rate": "30/1",
        }
        with self.assertRaises(SystemExit):
            stabilizer.parse_probe_json(self._json([stream]))

    def test_malformed_json_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            stabilizer.parse_probe_json("{no json")

    def test_non_dict_metadata_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            stabilizer.parse_probe_json(json.dumps(["streams"]))


class ModuleStructureTests(unittest.TestCase):
    def test_entry_reexports_public_api_and_core_modules_exist(self):
        import importlib

        core = importlib.import_module("eclipse_stabilizer_core")
        for module_name in ("constants", "video", "geometry", "profile", "tracking", "analysis", "render"):
            importlib.import_module(f"eclipse_stabilizer_core.{module_name}")
        for name in (
            "VideoInfo",
            "detect_limb",
            "apply_profile_discards",
            "load_profile",
            "solve_tracking",
            "classify_clipping",
            "classify_content",
            "classify_centering",
            "classify_regime",
            "robust_path_solution",
            "validate_preview",
            "parse_frame_spec",
            "effective_selection",
            "write_debug",
            "export_video",
            "EDGE_BITS",
            "EDGE_NAMES",
            "CONTENT_USABLE",
            "CONTENT_CLIPPED",
            "CONTENT_UNCERTAIN",
            "CENTER_RELIABLE",
            "CENTER_RECONSTRUCTED",
            "CENTER_UNRESOLVED",
            "REGIME_LIMBO",
            "REGIME_TRANSIENT",
            "REGIME_HORIZON",
            "build_parser",
            "command_inspect",
            "command_analyze",
            "command_preview",
            "command_export",
            "main",
            "load_or_analyze",
        ):
            self.assertTrue(hasattr(stabilizer, name), name)
        for removed in (
            "apply_timed_repairs",
            "visible_content_keep",
            "smooth_horizon_visible_jitter",
            "repair_visible_excursion",
        ):
            self.assertFalse(hasattr(stabilizer, removed), removed)
        self.assertEqual(core.ROOT, Path(__file__).resolve().parent)


class PhaseThreePortabilityTests(unittest.TestCase):
    """Phase 3: executable resolution, capability checks, exact-count
    verification and the exact-total iterator contract.  Unit cases run with
    mocks/env so they never depend on the system; integration cases run the
    resolved FFmpeg/ffprobe from PATH (skipping when unavailable)."""

    def setUp(self):
        stabilizer.reset_executable_cache()
        self.addCleanup(stabilizer.reset_executable_cache)

    def _require_ffmpeg(self):
        try:
            return stabilizer.resolve_ffmpeg()
        except SystemExit:
            self.skipTest("ffmpeg no disponible")

    # --- Executable resolution -------------------------------------------------

    def test_ffmpeg_resolution_priority_explicit_path_imageio(self):
        explicit = r"C:\explicit\ffmpeg.exe"
        on_path = r"C:\path\ffmpeg.exe"
        bundled = r"C:\imageio\ffmpeg.exe"
        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=True):
            with mock.patch.dict(os.environ, {"FFMPEG": explicit}):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=on_path):
                    self.assertEqual(stabilizer.resolve_ffmpeg(), explicit)
            stabilizer.reset_executable_cache()
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=on_path):
                    self.assertEqual(stabilizer.resolve_ffmpeg(), on_path)
            stabilizer.reset_executable_cache()
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    with mock.patch.dict(
                        sys.modules,
                        {"imageio_ffmpeg": mock.Mock(get_ffmpeg_exe=mock.Mock(return_value=bundled))},
                    ):
                        self.assertEqual(stabilizer.resolve_ffmpeg(), bundled)

    def test_invalid_explicit_ffmpeg_does_not_fall_back(self):
        explicit = r"C:\missing\ffmpeg.exe"
        with mock.patch(
            "eclipse_stabilizer_core.video._executable_responds",
            side_effect=lambda exe: exe != explicit,
        ):
            with mock.patch.dict(os.environ, {"FFMPEG": explicit}):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=r"C:\path\ffmpeg.exe"):
                    with mock.patch.dict(
                        sys.modules,
                        {"imageio_ffmpeg": mock.Mock(get_ffmpeg_exe=mock.Mock(return_value=r"C:\imageio\ffmpeg.exe"))},
                    ):
                        with self.assertRaises(SystemExit) as ctx:
                            stabilizer.resolve_ffmpeg()
        self.assertIn("FFMPEG", str(ctx.exception))
        self.assertIn("no responde", str(ctx.exception))

    def test_no_ffmpeg_anywhere_raises_clear_error(self):
        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=False):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    with mock.patch.dict(sys.modules, {"imageio_ffmpeg": None}):
                        with self.assertRaises(SystemExit) as ctx:
                            stabilizer.resolve_ffmpeg()
        self.assertIn("No se encontró un FFmpeg válido", str(ctx.exception))

    def test_ffprobe_resolution_priority_explicit_path_none(self):
        explicit = r"C:\explicit\ffprobe.exe"
        on_path = r"C:\path\ffprobe.exe"
        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=True):
            with mock.patch.dict(os.environ, {"FFPROBE": explicit}):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=on_path):
                    self.assertEqual(stabilizer.resolve_ffprobe(), explicit)
            stabilizer.reset_executable_cache()
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=on_path):
                    self.assertEqual(stabilizer.resolve_ffprobe(), on_path)
            stabilizer.reset_executable_cache()
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    self.assertIsNone(stabilizer.resolve_ffprobe())

    def test_invalid_explicit_ffprobe_does_not_fall_back(self):
        explicit = r"C:\missing\ffprobe.exe"
        with mock.patch(
            "eclipse_stabilizer_core.video._executable_responds",
            side_effect=lambda exe: exe != explicit,
        ):
            with mock.patch.dict(os.environ, {"FFPROBE": explicit}):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=r"C:\path\ffprobe.exe"):
                    with self.assertRaises(SystemExit) as ctx:
                        stabilizer.resolve_ffprobe()
        self.assertIn("FFPROBE", str(ctx.exception))
        self.assertIn("no responde", str(ctx.exception))

    # --- Capabilities ----------------------------------------------------------

    def test_capabilities_decode_do_not_require_encoder(self):
        calls = []
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module, "_check_raw_pipeline", side_effect=lambda *a, **k: calls.append("pipeline")):
                with mock.patch.object(video_module, "_check_libx264", side_effect=lambda *a, **k: calls.append("encoder")):
                    exe = stabilizer.ensure_capabilities(need_encoder=False)
        self.assertEqual(exe, r"C:\resolved\ffmpeg.exe")
        self.assertEqual(calls.count("pipeline"), 2)
        self.assertEqual(calls.count("encoder"), 0)

    def test_capabilities_encoder_requires_libx264(self):
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module, "_check_raw_pipeline"):
                with mock.patch.object(video_module, "_check_libx264", side_effect=SystemExit("sin libx264")):
                    stabilizer.reset_executable_cache()
                    with self.assertRaises(SystemExit) as ctx:
                        stabilizer.ensure_capabilities(need_encoder=True)
        self.assertIn("sin libx264", str(ctx.exception))

    def test_binary_without_decode_capability_is_rejected(self):
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module, "_check_raw_pipeline", side_effect=SystemExit("sin decode")):
                with self.assertRaises(SystemExit) as ctx:
                    stabilizer.ensure_capabilities(need_encoder=False)
        self.assertIn("sin decode", str(ctx.exception))

    def test_capabilities_real_system_ffmpeg(self):
        self._require_ffmpeg()
        exe = stabilizer.ensure_capabilities(need_encoder=True)
        self.assertEqual(exe, stabilizer.resolve_ffmpeg())
        self.assertTrue(stabilizer.ffmpeg_version(exe).startswith("ffmpeg version"))

    # --- VideoInfo / probe -----------------------------------------------------

    def test_videoinfo_defaults(self):
        info = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0)
        self.assertEqual(info.probe_source, "ffprobe")
        self.assertTrue(info.frame_count_exact)

    def test_parse_probe_json_sets_frame_count_exact(self):
        exact_json = json.dumps(
            {
                "streams": [
                    {
                        "width": 240,
                        "height": 360,
                        "avg_frame_rate": "30/1",
                        "r_frame_rate": "30/1",
                        "nb_frames": "30",
                        "duration": "1.0",
                    }
                ],
                "format": {"duration": "1.0"},
            }
        )
        info, estimated = stabilizer.parse_probe_json(exact_json)
        self.assertFalse(estimated)
        self.assertTrue(info.frame_count_exact)
        self.assertEqual(info.probe_source, "ffprobe")

        estimated_json = json.dumps(
            {
                "streams": [
                    {
                        "width": 240,
                        "height": 360,
                        "avg_frame_rate": "30/1",
                        "r_frame_rate": "30/1",
                        "nb_frames": "N/A",
                        "duration": "1.0",
                    }
                ],
                "format": {"duration": "1.0"},
            }
        )
        info2, est2 = stabilizer.parse_probe_json(estimated_json)
        self.assertTrue(est2)
        self.assertFalse(info2.frame_count_exact)

    def test_probe_video_ffprobe_marks_exact(self):
        if shutil.which("ffprobe") is None:
            self.skipTest("ffprobe no disponible")
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, 20)
            info = stabilizer.probe_video(video)
            self.assertTrue(info.frame_count_exact)
            self.assertEqual(info.probe_source, "ffprobe")
            self.assertEqual(info.frames, 20)

    def test_probe_video_uses_opencv_when_no_ffprobe(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 240,
            cv2.CAP_PROP_FRAME_HEIGHT: 360,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 50,
        }.get(prop, 0)
        with mock.patch.object(video_module, "resolve_ffprobe", return_value=None):
            with mock.patch.object(video_module.cv2, "VideoCapture", return_value=cap):
                info = stabilizer.probe_video(Path("v.mp4"))
        self.assertFalse(info.frame_count_exact)
        self.assertEqual(info.probe_source, "opencv")
        self.assertEqual((info.width, info.height, info.frames), (240, 360, 50))

    def test_probe_opencv_invalid_is_rejected(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0
        with mock.patch.object(video_module.cv2, "VideoCapture", return_value=cap):
            with self.assertRaises(SystemExit):
                stabilizer.probe_opencv(Path("v.mp4"))
        closed = mock.Mock()
        closed.isOpened.return_value = False
        with mock.patch.object(video_module.cv2, "VideoCapture", return_value=closed):
            with self.assertRaises(SystemExit):
                stabilizer.probe_opencv(Path("v.mp4"))

    def test_print_video_info_marks_provisional(self):
        info = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0, probe_source="opencv", frame_count_exact=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stabilizer.print_video_info(Path("v.mp4"), info)
        self.assertIn("provisional", buffer.getvalue())

    def test_print_video_info_no_marker_after_exact_verification(self):
        verified = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0, probe_source="opencv", frame_count_exact=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stabilizer.print_video_info(Path("v.mp4"), verified)
        self.assertNotIn("provisional", buffer.getvalue())

    # --- vfrdet parser / strict verification -----------------------------------

    def test_parse_vfrdet_cfr(self):
        parsed = stabilizer.parse_vfrdet_output(
            "[Parsed_vfrdet_0 @ x] VFR:nan (0/0)\n"
            "[Parsed_vfrdet_0 @ y] VFR:0.000000 (0/2399)\n"
        )
        self.assertEqual(parsed["vfr"], 0)
        self.assertEqual(parsed["frames"], 2400)

    def test_parse_vfrdet_cfr_old_format(self):
        parsed = stabilizer.parse_vfrdet_output("VFR:0.000000 (0/29) CFR:1.000000")
        self.assertEqual(parsed["vfr"], 0)
        self.assertEqual(parsed["frames"], 30)

    def test_parse_vfrdet_vfr(self):
        parsed = stabilizer.parse_vfrdet_output("VFR:0.655172 (19/10) min: 33 max: 34 avg: 33")
        self.assertEqual(parsed["vfr"], 19)
        self.assertEqual(parsed["frames"], 30)

    def test_parse_vfrdet_inconclusive(self):
        self.assertIsNone(stabilizer.parse_vfrdet_output(""))
        self.assertIsNone(stabilizer.parse_vfrdet_output("VFR:nan (0/0)"))
        self.assertIsNone(stabilizer.parse_vfrdet_output("sin medición vfrdet"))

    def test_verify_exact_cfr_accepts_cfr_synthetic(self):
        self._require_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, 24)
            provisional = stabilizer.VideoInfo(240, 360, 24, 30.0, 24 / 30.0, probe_source="opencv", frame_count_exact=False)
            info = stabilizer.verify_exact_cfr(video, provisional)
            self.assertTrue(info.frame_count_exact)
            self.assertEqual(info.frames, 24)

    def test_verify_exact_cfr_rejects_vfr_simulated(self):
        provisional = stabilizer.VideoInfo(64, 64, 30, 30.0, 1.0, probe_source="opencv", frame_count_exact=False)
        with mock.patch.object(video_module, "run_capture", return_value="VFR:0.655172 (19/10) min: 33 max: 34 avg: 33"):
            with self.assertRaises(SystemExit) as ctx:
                stabilizer.verify_exact_cfr(Path("v.mkv"), provisional)
        self.assertIn("cadencia variable", str(ctx.exception))

    def test_verify_exact_cfr_count_mismatch_rejected(self):
        provisional = stabilizer.VideoInfo(240, 360, 40, 30.0, 40 / 30.0, probe_source="opencv", frame_count_exact=False)
        with mock.patch.object(video_module, "run_capture", return_value="VFR:0.000000 (0/29)"):
            with self.assertRaises(SystemExit) as ctx:
                stabilizer.verify_exact_cfr(Path("v.mp4"), provisional)
        self.assertIn("difiere del estimado", str(ctx.exception))

    def test_verify_exact_cfr_inconclusive_rejected(self):
        provisional = stabilizer.VideoInfo(240, 360, 40, 30.0, 40 / 30.0, probe_source="opencv", frame_count_exact=False)
        with mock.patch.object(video_module, "run_capture", return_value=""):
            with self.assertRaises(SystemExit) as ctx:
                stabilizer.verify_exact_cfr(Path("v.mp4"), provisional)
        self.assertIn("no pudo concluir", str(ctx.exception))

    def _write_vfr_fixture(self, tmp):
        exe = self._require_ffmpeg()
        path = Path(tmp) / "vfr.mkv"
        command = [
            exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30:duration=1",
            "-vf",
            "setpts='if(eq(mod(N,15),0),PTS+0.5*TB,PTS)'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "vfr",
            "-f",
            "matroska",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return path

    def test_verify_exact_cfr_rejects_real_vfr_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_vfr_fixture(tmp)
            if path is None:
                self.skipTest("este FFmpeg no genera el fixture VFR; el rechazo se cubre con parser/simulado")
            exe = self._require_ffmpeg()
            raw = subprocess.run(
                [exe, "-hide_banner", "-loglevel", "info", "-i", str(path), "-vf", "vfrdet", "-an", "-f", "null", "-"],
                capture_output=True,
                text=True,
            ).stderr
            parsed = stabilizer.parse_vfrdet_output(raw)
            if parsed is None or parsed["vfr"] == 0:
                self.skipTest("este FFmpeg no produce VFR detectable en el fixture; el rechazo se cubre con parser/simulado")
            provisional = stabilizer.VideoInfo(64, 64, parsed["frames"], 30.0, parsed["frames"] / 30.0, probe_source="opencv", frame_count_exact=False)
            with self.assertRaises(SystemExit) as ctx:
                stabilizer.verify_exact_cfr(path, provisional)
            self.assertIn("cadencia variable", str(ctx.exception))

    # --- Cache identity --------------------------------------------------------

    def test_cache_identity_uses_frame_count_exact_ignores_probe_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = tmp / "a.mp4"
            v1.write_bytes(b"x" * 1024)

            def build(info):
                return stabilizer.build_cache_identity(v1, info, 270, 480, "auto", None, None, 0.18)

            exact = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0, probe_source="ffprobe", frame_count_exact=True)
            same_opencv = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0, probe_source="opencv", frame_count_exact=True)
            provisional = stabilizer.VideoInfo(240, 360, 100, 30.0, 100 / 30.0, probe_source="opencv", frame_count_exact=False)
            id_exact = build(exact)
            self.assertEqual(build(same_opencv), id_exact)
            self.assertNotEqual(build(provisional), id_exact)
            self.assertNotIn("probe_source", id_exact)
            self.assertTrue(id_exact["frame_count_exact"])

    # --- Exact-total iterators ------------------------------------------------

    def test_exact_total_sentinel_under_over_and_exact(self):
        self._require_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, 30)
            gray = list(stabilizer.iter_ffmpeg_gray(video, 120, 180, 30, exact_total=True))
            self.assertEqual(len(gray), 30)
            bgr = list(stabilizer.iter_ffmpeg_bgr(video, 120, 180, 30, exact_total=True))
            self.assertEqual(len(bgr), 30)
            with self.assertRaises(SystemExit) as ctx:
                list(stabilizer.iter_ffmpeg_gray(video, 120, 180, 29, exact_total=True))
            self.assertIn("centinela", str(ctx.exception))
            with self.assertRaises(SystemExit) as ctx:
                list(stabilizer.iter_ffmpeg_bgr(video, 120, 180, 33, exact_total=True))
            self.assertIn("demasiado alto", str(ctx.exception))

    def test_bounded_prefix_has_no_false_sentinel(self):
        self._require_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, 30)
            frames = list(stabilizer.iter_ffmpeg_bgr(video, 120, 180, 5))
            self.assertEqual(len(frames), 5)
            gray = list(stabilizer.iter_ffmpeg_gray(video, 120, 180, 7))
            self.assertEqual(len(gray), 7)

    def test_export_prefix_does_not_trigger_sentinel(self):
        self._require_ffmpeg()
        n = 30
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(tmp, n)
            info = stabilizer.probe_video(video)
            keep = np.ones(n, bool)
            keep[3] = False
            analysis = {
                "analysis_width": np.array([info.width]),
                "analysis_height": np.array([info.height]),
                "trim_end": np.array([9]),
                "keep": keep,
                "center": np.tile([info.width / 2.0, info.height / 2.0], (n, 1)),
            }
            out = Path(tmp) / "out"
            dest = out / "stabilized.mp4"
            stabilizer.export_video(
                video, info, analysis, dest, info.width, info.height,
                speed=1.0, crf=20, preset="ultrafast", debug_overlay=False,
            )
            probe = stabilizer.probe_video(dest)
            self.assertEqual(probe.frames, 9)

    # --- Commands use the resolved executable ----------------------------------

    def test_iterators_use_resolved_executable(self):
        captured = []

        def fake_popen(command, **kwargs):
            captured.append(command)
            proc = mock.Mock()
            proc.stdout.read.return_value = b""
            proc.stdout.close = mock.Mock()
            proc.stderr.read.return_value = b""
            proc.stderr.close = mock.Mock()
            proc.returncode = 0
            proc.wait.return_value = None
            return proc

        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module.subprocess, "Popen", side_effect=fake_popen):
                list(stabilizer.iter_ffmpeg_gray(Path("v.mp4"), 120, 180, 5))
                list(stabilizer.iter_ffmpeg_bgr(Path("v.mp4"), 120, 180, 5))
        self.assertEqual(len(captured), 2)
        gray_cmd, bgr_cmd = captured
        for cmd in (gray_cmd, bgr_cmd):
            self.assertEqual(cmd[0], r"C:\resolved\ffmpeg.exe")
        self.assertIn("scale=120:180:flags=area,format=gray", gray_cmd)
        self.assertIn("scale=120:180:flags=lanczos", bgr_cmd)

    def test_export_command_uses_resolved_executable_and_keeps_params(self):
        analysis = synthetic_analysis(10)
        analysis.update(stabilizer.solve_tracking(analysis, 0.18))
        info = stabilizer.VideoInfo(240, 360, 10, 30.0, 10 / 30.0)
        captured = {}

        def fake_frames(video, width, height, frames, *, exact_total=False):
            for i in range(frames):
                yield i, np.zeros((height, width, 3), np.uint8)

        class FakeProc:
            def __init__(self, command, **kwargs):
                captured["command"] = command
                Path(command[-1]).write_bytes(b"fake mp4")
                self.stdin = mock.Mock()
                self.stderr = mock.Mock()
                self.stderr.read.return_value = b""
                self.returncode = 0

            def wait(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "stabilized.mp4"
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=fake_frames):
                    with mock.patch.object(render_module.subprocess, "Popen", FakeProc):
                        stabilizer.export_video(
                            Path("v.mp4"), info, analysis, dest, 240, 360,
                            speed=1.0, crf=18, preset="fast", debug_overlay=False, threads=2,
                        )
            # The encoder wrote to a unique sibling temp keeping the .mp4 suffix,
            # never to the destination itself; publication replaced the destination.
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"fake mp4")
            self.assertNotEqual(captured["command"][-1], str(dest))
            self.assertTrue(captured["command"][-1].endswith(".mp4"))
            self.assertNotIn(str(dest), captured["command"])
        cmd = captured["command"]
        self.assertEqual(cmd[0], r"C:\resolved\ffmpeg.exe")
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("+faststart", cmd)
        self.assertIn("-an", cmd)
        self.assertEqual(cmd[cmd.index("-preset") + 1], "fast")
        self.assertEqual(cmd[cmd.index("-crf") + 1], "18")
        self.assertEqual(cmd[cmd.index("-threads") + 1], "2")

    def test_write_debug_uses_resolved_executable(self):
        n = 4
        analysis = synthetic_analysis(n)
        analysis.update(stabilizer.solve_tracking(analysis, 0.18))
        info = stabilizer.VideoInfo(240, 360, n, 30.0, n / 30.0)
        captured = {}

        def fake_frames(video, width, height, frames, *, exact_total=False):
            for i in range(frames):
                yield i, np.zeros((height, width, 3), np.uint8)

        class FakeProc:
            def __init__(self, command, **kwargs):
                captured["command"] = command
                Path(command[-1]).write_bytes(b"fake review")
                self.stdin = mock.Mock()
                self.stderr = mock.Mock()
                self.stderr.read.return_value = b""
                self.returncode = 0

            def wait(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=fake_frames):
                    with mock.patch.object(render_module, "_estimate_jpeg_bytes", return_value=0.0):
                        with mock.patch.object(render_module.subprocess, "Popen", FakeProc):
                            stabilizer.write_debug(Path("v.mp4"), info, analysis, out, debug_width=320, max_images=10000)
            # The review was published atomically to debug/review.mp4.
            self.assertEqual((out / "debug" / "review.mp4").read_bytes(), b"fake review")
            self.assertNotEqual(captured["command"][-1], str(out / "debug" / "review.mp4"))
        cmd = captured["command"]
        self.assertEqual(cmd[0], r"C:\resolved\ffmpeg.exe")
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)

    def test_imageio_env_respected_in_last_fallback(self):
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            self.skipTest("imageio-ffmpeg no instalado")
        custom = r"C:\custom\ffmpeg.exe"
        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=True):
            with mock.patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": custom}):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    self.assertEqual(stabilizer.resolve_ffmpeg(), custom)

    def test_packaged_fallback_without_path(self):
        """End-to-end fallback: with PATH stripped of the system ffmpeg/ffprobe
        directories and FFMPEG/FFPROBE unset, the subprocess must resolve and
        use the imageio-ffmpeg bundled binary (which also supports libx264)."""
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest("imageio-ffmpeg no instalado")
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if not bundled or not os.path.exists(bundled):
            self.skipTest("imageio-ffmpeg sin binario empaquetado válido")
        blocked = set()
        for name in ("ffmpeg", "ffprobe"):
            found = shutil.which(name)
            if found:
                blocked.add(os.path.normcase(os.path.abspath(os.path.dirname(found))))
        env = dict(os.environ)
        env.pop("FFMPEG", None)
        env.pop("FFPROBE", None)
        env["PATH"] = os.pathsep.join(
            d for d in env.get("PATH", "").split(os.pathsep)
            if d and os.path.normcase(os.path.abspath(d)) not in blocked
        )
        code = (
            "from eclipse_stabilizer_core.video import resolve_ffmpeg, resolve_ffprobe, ensure_capabilities; "
            "print(resolve_ffmpeg()); print(resolve_ffprobe()); print(ensure_capabilities(need_encoder=True))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[0].strip(), bundled)
        self.assertEqual(lines[1].strip(), "None")
        self.assertEqual(lines[2].strip(), bundled)

    # --- Correction: capture_bytes / quick-check hardening -------------------

    def test_capture_bytes_timeout_raises_clear_error(self):
        with mock.patch.object(
            video_module.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg", "-version"], timeout=30),
        ):
            with self.assertRaises(SystemExit) as ctx:
                video_module._capture_bytes(["ffmpeg", "-version"])
        self.assertIn("30 s", str(ctx.exception))
        self.assertIn("agotó el tiempo", str(ctx.exception))

    def test_capture_bytes_nonzero_returncode_raises_clear_error(self):
        result = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout=b"", stderr=b"boom"
        )
        with mock.patch.object(video_module.subprocess, "run", return_value=result):
            with self.assertRaises(SystemExit) as ctx:
                video_module._capture_bytes(["ffmpeg", "-i", "x"])
        self.assertIn("código 1", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_capture_bytes_returns_stdout_only(self):
        result = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout=b"\x00" * 16, stderr=b"warn"
        )
        with mock.patch.object(video_module.subprocess, "run", return_value=result):
            data = video_module._capture_bytes(["ffmpeg", "-i", "x"])
        self.assertEqual(data, b"\x00" * 16)

    def test_check_raw_pipeline_ignores_stderr(self):
        good = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout=b"\x00" * (16 * 16), stderr=b"warning"
        )
        with mock.patch.object(video_module.subprocess, "run", return_value=good):
            video_module._check_raw_pipeline("ffmpeg", "scale=16:16:flags=area,format=gray", "gray", 16 * 16)
        bad = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout=b"\x00" * 100, stderr=b"warning"
        )
        with mock.patch.object(video_module.subprocess, "run", return_value=bad):
            with self.assertRaises(SystemExit) as ctx:
                video_module._check_raw_pipeline("ffmpeg", "scale=16:16:flags=area,format=gray", "gray", 16 * 16)
        self.assertIn("formato raw esperado", str(ctx.exception))

    # --- Correction: imageio exceptions are swallowed cleanly ----------------

    def test_imageio_import_exception_is_caught(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise RuntimeError("imageio roto")
            return real_import(name, *args, **kwargs)

        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=False):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    with mock.patch("builtins.__import__", side_effect=fake_import):
                        with self.assertRaises(SystemExit) as ctx:
                            stabilizer.resolve_ffmpeg()
        self.assertIn("No se encontró un FFmpeg válido", str(ctx.exception))

    def test_imageio_get_ffmpeg_exe_exception_is_caught(self):
        fake = mock.Mock(get_ffmpeg_exe=mock.Mock(side_effect=RuntimeError("boom")))
        with mock.patch("eclipse_stabilizer_core.video._executable_responds", return_value=False):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("eclipse_stabilizer_core.video.shutil.which", return_value=None):
                    with mock.patch.dict(sys.modules, {"imageio_ffmpeg": fake}):
                        with self.assertRaises(SystemExit) as ctx:
                            stabilizer.resolve_ffmpeg()
        self.assertIn("No se encontró un FFmpeg válido", str(ctx.exception))

    # --- Correction: vfrdet filter missing is actionable ---------------------

    def test_verify_exact_cfr_missing_vfrdet_filter_actionable(self):
        provisional = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0, probe_source="opencv", frame_count_exact=False)
        with mock.patch.object(
            video_module, "run_capture", side_effect=SystemExit("No such filter: 'vfrdet'")
        ):
            with self.assertRaises(SystemExit) as ctx:
                stabilizer.verify_exact_cfr(Path("v.mp4"), provisional)
        self.assertIn("vfrdet", str(ctx.exception))
        self.assertIn("FFMPEG", str(ctx.exception))

    # --- Correction: sentinel is never masked by the finally returncode ------

    def test_sentinel_not_masked_by_ffmpeg_returncode(self):
        width, height = 120, 180
        frame = b"\x00" * (width * height)

        def fake_popen(command, **kwargs):
            proc = mock.Mock()
            proc.stdout.read.side_effect = [frame, frame]  # requested = frames + 1
            proc.stdout.close = mock.Mock()
            proc.stderr.read.return_value = b"error"
            proc.stderr.close = mock.Mock()
            proc.returncode = 1
            proc.wait.return_value = None
            return proc

        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module.subprocess, "Popen", side_effect=fake_popen):
                with self.assertRaises(SystemExit) as ctx:
                    list(stabilizer.iter_ffmpeg_gray(Path("v.mp4"), width, height, 1, exact_total=True))
        self.assertIn("centinela", str(ctx.exception))
        self.assertNotIn("FFmpeg entregó", str(ctx.exception))

    # --- Correction: capability cache is per executable ----------------------

    def test_capability_cache_per_exe_avoids_rerunning_raw_pipelines(self):
        calls = []
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module, "_check_raw_pipeline", side_effect=lambda *a, **k: calls.append("pipeline")):
                with mock.patch.object(video_module, "_check_libx264", side_effect=lambda *a, **k: calls.append("encoder")):
                    stabilizer.ensure_capabilities(need_encoder=False)
                    stabilizer.ensure_capabilities(need_encoder=True)
        self.assertEqual(calls.count("pipeline"), 2)
        self.assertEqual(calls.count("encoder"), 1)

    # --- Correction: main banner order / probe source / no dir before verify -

    def test_main_banner_shows_executables_and_probe_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            captured = io.StringIO()
            info = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0)
            argv = ["eclipse_stabilizer.py", "--out", str(out_dir), "inspect", "video.mp4"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(stabilizer, "find_video", return_value=Path("video.mp4")):
                    with mock.patch.object(stabilizer, "load_profile", return_value=None):
                        with mock.patch.object(stabilizer, "probe_video", return_value=info):
                            with mock.patch.object(stabilizer, "resolve_output_dir", return_value=out_dir):
                                with mock.patch.object(stabilizer, "ensure_capabilities", return_value=r"C:\ff\ffmpeg.exe"):
                                    with mock.patch.object(stabilizer, "ffmpeg_version", return_value="ffmpeg version 8.1-test"):
                                        with mock.patch.object(stabilizer, "resolve_ffprobe", return_value=r"C:\ff\ffprobe.exe"):
                                            with mock.patch.object(stabilizer, "build_cache_identity", return_value={}):
                                                with mock.patch.object(stabilizer, "cache_status", return_value=("missing", [])):
                                                    with mock.patch.object(stabilizer, "command_inspect", side_effect=lambda *a, **k: None):
                                                        with contextlib.redirect_stdout(captured):
                                                            stabilizer.main()
            text = captured.getvalue()
            self.assertIn(r"C:\ff\ffmpeg.exe", text)
            self.assertIn("ffmpeg version 8.1-test", text)
            self.assertIn(r"C:\ff\ffprobe.exe", text)

    def test_main_banner_shows_opencv_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            captured = io.StringIO()
            info = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0)
            argv = ["eclipse_stabilizer.py", "--out", str(out_dir), "inspect", "video.mp4"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(stabilizer, "find_video", return_value=Path("video.mp4")):
                    with mock.patch.object(stabilizer, "load_profile", return_value=None):
                        with mock.patch.object(stabilizer, "probe_video", return_value=info):
                            with mock.patch.object(stabilizer, "resolve_output_dir", return_value=out_dir):
                                with mock.patch.object(stabilizer, "ensure_capabilities", return_value="exe"):
                                    with mock.patch.object(stabilizer, "ffmpeg_version", return_value="v"):
                                        with mock.patch.object(stabilizer, "resolve_ffprobe", return_value=None):
                                            with mock.patch.object(stabilizer, "build_cache_identity", return_value={}):
                                                with mock.patch.object(stabilizer, "cache_status", return_value=("missing", [])):
                                                    with mock.patch.object(stabilizer, "command_inspect", side_effect=lambda *a, **k: None):
                                                        with contextlib.redirect_stdout(captured):
                                                            stabilizer.main()
            self.assertIn("OpenCV fallback", captured.getvalue())

    def test_main_banner_distinguishes_refreshable_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            captured = io.StringIO()
            info = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0)
            argv = ["eclipse_stabilizer.py", "--out", str(out_dir), "inspect", "video.mp4"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(stabilizer, "find_video", return_value=Path("video.mp4")):
                    with mock.patch.object(stabilizer, "load_profile", return_value=None):
                        with mock.patch.object(stabilizer, "probe_video", return_value=info):
                            with mock.patch.object(stabilizer, "resolve_output_dir", return_value=out_dir):
                                with mock.patch.object(stabilizer, "ensure_capabilities", return_value="exe"):
                                    with mock.patch.object(stabilizer, "ffmpeg_version", return_value="v"):
                                        with mock.patch.object(stabilizer, "resolve_ffprobe", return_value=None):
                                            with mock.patch.object(stabilizer, "build_cache_identity", return_value={}):
                                                with mock.patch.object(
                                                    stabilizer, "cache_status",
                                                    return_value=("refreshable", ["'contrast_version' difiere: caché=2, esperado=3"]),
                                                ):
                                                    with mock.patch.object(stabilizer, "command_inspect", side_effect=lambda *a, **k: None):
                                                        with contextlib.redirect_stdout(captured):
                                                            stabilizer.main()
            self.assertIn("Caché       : refrescable (solo contraste)", captured.getvalue())

    def test_main_checks_capabilities_before_vfrdet_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            order = []
            provisional = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0, probe_source="opencv", frame_count_exact=False)

            def cap(need_encoder=False):
                order.append("capabilities")
                return "exe"

            def verify(video, info):
                order.append("verify")
                return stabilizer.VideoInfo(
                    info.width, info.height, info.frames, info.fps, info.duration,
                    probe_source=info.probe_source, frame_count_exact=True,
                )

            argv = ["eclipse_stabilizer.py", "--out", str(out_dir), "analyze", "video.mp4"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(stabilizer, "find_video", return_value=Path("video.mp4")):
                    with mock.patch.object(stabilizer, "load_profile", return_value=None):
                        with mock.patch.object(stabilizer, "probe_video", return_value=provisional):
                            with mock.patch.object(stabilizer, "resolve_output_dir", return_value=out_dir):
                                with mock.patch.object(stabilizer, "ensure_capabilities", side_effect=cap):
                                    with mock.patch.object(stabilizer, "ffmpeg_version", return_value="v"):
                                        with mock.patch.object(stabilizer, "resolve_ffprobe", return_value=None):
                                            with mock.patch.object(stabilizer, "verify_exact_cfr", side_effect=verify):
                                                with mock.patch.object(stabilizer, "build_cache_identity", return_value={}):
                                                    with mock.patch.object(stabilizer, "cache_status", return_value=("missing", [])):
                                                        with mock.patch.object(stabilizer, "command_analyze", side_effect=lambda *a, **k: None):
                                                            with contextlib.redirect_stdout(io.StringIO()):
                                                                stabilizer.main()
            self.assertEqual(order, ["capabilities", "verify"])

    def test_main_aborts_before_creating_output_dir_on_verification_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            provisional = stabilizer.VideoInfo(240, 360, 30, 30.0, 1.0, probe_source="opencv", frame_count_exact=False)
            argv = ["eclipse_stabilizer.py", "--out", str(out_dir), "analyze", "video.mp4"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(stabilizer, "find_video", return_value=Path("video.mp4")):
                    with mock.patch.object(stabilizer, "load_profile", return_value=None):
                        with mock.patch.object(stabilizer, "probe_video", return_value=provisional):
                            with mock.patch.object(stabilizer, "resolve_output_dir", return_value=out_dir):
                                with mock.patch.object(stabilizer, "ensure_capabilities", return_value="exe"):
                                    with mock.patch.object(stabilizer, "ffmpeg_version", return_value="v"):
                                        with mock.patch.object(stabilizer, "resolve_ffprobe", return_value=None):
                                            with mock.patch.object(stabilizer, "verify_exact_cfr", side_effect=SystemExit("VFR detectado")):
                                                with contextlib.redirect_stdout(io.StringIO()):
                                                    with self.assertRaises(SystemExit):
                                                        stabilizer.main()
            self.assertFalse(out_dir.exists())

    # --- Correction: run_capture tolerates bytes invalid for the UTF-8 codec ---

    def test_run_capture_invalid_bytes_do_not_propagate_unicode_decode_error(self):
        invalid = b"\xff"
        with self.assertRaises(UnicodeDecodeError):
            invalid.decode()
        with mock.patch.object(video_module.subprocess, "check_output", return_value=invalid):
            text = video_module.run_capture(["ffmpeg", "-version"])
        self.assertIn("\ufffd", text)

    # --- Correction: iterators drain stderr before wait (>64 KiB risk) ------

    def _drain_before_wait_proc(self, width, height, frames, *, color):
        size = width * height * (3 if color else 1)
        order = []
        payload = b"E" * (70 * 1024)
        proc = mock.Mock()
        proc.returncode = 0

        def read_stdout(n):
            order.append("stdout.read")
            return b"\x00" * size

        def read_stderr():
            order.append("stderr.read")
            return payload

        def do_wait():
            order.append("wait")

        def do_close_stderr():
            order.append("stderr.close")

        proc.stdout.read.side_effect = read_stdout
        proc.stderr.read.side_effect = read_stderr
        proc.wait.side_effect = do_wait
        proc.stdout.close = mock.Mock()
        proc.stderr.close = mock.Mock(side_effect=do_close_stderr)
        return proc, order, payload

    def test_iter_ffmpeg_gray_drains_stderr_before_wait(self):
        width, height = 120, 180
        proc, order, payload = self._drain_before_wait_proc(width, height, 1, color=False)
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module.subprocess, "Popen", return_value=proc):
                frames = list(stabilizer.iter_ffmpeg_gray(Path("v.mp4"), width, height, 1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(proc.stderr.read(), payload)
        self.assertGreater(len(payload), 64 * 1024)
        self.assertLess(order.index("stderr.read"), order.index("stderr.close"))
        self.assertLess(order.index("stderr.close"), order.index("wait"))

    def test_iter_ffmpeg_bgr_drains_stderr_before_wait(self):
        width, height = 120, 180
        proc, order, payload = self._drain_before_wait_proc(width, height, 1, color=True)
        with mock.patch.object(video_module, "resolve_ffmpeg", return_value=r"C:\resolved\ffmpeg.exe"):
            with mock.patch.object(video_module.subprocess, "Popen", return_value=proc):
                frames = list(stabilizer.iter_ffmpeg_bgr(Path("v.mp4"), width, height, 1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(proc.stderr.read(), payload)
        self.assertGreater(len(payload), 64 * 1024)
        self.assertLess(order.index("stderr.read"), order.index("stderr.close"))
        self.assertLess(order.index("stderr.close"), order.index("wait"))


class AtomicVideoPublicationTests(unittest.TestCase):
    """Phase 3 (part 2): atomic publication of export_video and debug review.

    All cases use mocks plus temporary directories (never real video data): a fake
    encoder writes the sibling temp that FFmpeg would have produced, and the
    assertions check that the destination is only replaced on full success and
    that no temp file survives any failure path."""

    def _analysis(self, n):
        analysis = synthetic_analysis(n)
        analysis.update(stabilizer.solve_tracking(analysis, 0.18))
        return analysis

    def _info(self, n):
        return stabilizer.VideoInfo(240, 360, n, 30.0, n / 30.0)

    def _iter_frames(self, count):
        def fake_frames(video, width, height, frames, *, exact_total=False):
            for i in range(min(count, frames)):
                yield i, np.zeros((height, width, 3), np.uint8)

        return fake_frames

    def _fake_encoder(self, captured, returncode=0, stderr=b""):
        class FakeProc:
            def __init__(self, command, **kwargs):
                captured["command"] = command
                Path(command[-1]).write_bytes(b"temporal")
                self.stdin = mock.Mock()
                self.stderr = mock.Mock()
                self.stderr.read.return_value = stderr
                self.returncode = None
                self._final = returncode

            def wait(self):
                captured["waited"] = True
                self.returncode = self._final

        return FakeProc

    def _temps(self, directory, stem, suffix=".mp4"):
        return list(Path(directory).glob(f"{stem}.*{suffix}"))

    def _export(self, analysis, info, dest, **overrides):
        args = dict(
            video=Path("v.mp4"),
            info=info,
            analysis=analysis,
            destination=dest,
            width=240,
            height=360,
            speed=1.0,
            crf=20,
            preset="ultrafast",
            debug_overlay=False,
        )
        args.update(overrides)
        stabilizer.export_video(**args)

    def test_export_success_publishes_and_leaves_no_temp(self):
        n = 10
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                        with contextlib.redirect_stdout(io.StringIO()):
                            self._export(self._analysis(n), self._info(n), dest)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"temporal")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])
            self.assertTrue(captured["waited"])
            self.assertTrue(captured["command"][-1].endswith(".mp4"))
            self.assertNotEqual(captured["command"][-1], str(dest))

    def test_export_short_count_preserves_previous_destination_and_cleans_temp(self):
        n = 10
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(6)):
                    with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                        with self.assertRaises(SystemExit) as ctx:
                            self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("incompleta", str(ctx.exception))
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_export_encoder_failure_cleans_temp_and_preserves_destination(self):
        n = 10
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(
                        render_module.subprocess, "Popen",
                        self._fake_encoder(captured, returncode=1, stderr=b"error del codificador"),
                    ):
                        with self.assertRaises(SystemExit) as ctx:
                            self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("error del codificador", str(ctx.exception))
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_export_broken_pipe_cleans_temp_and_reaps_encoder(self):
        n = 10
        captured = {}

        class BrokenPipeProc:
            def __init__(self, command, **kwargs):
                captured["command"] = command
                Path(command[-1]).write_bytes(b"temporal")
                self.stdin = mock.Mock()
                self.stdin.write.side_effect = BrokenPipeError()
                self.stdin.close.side_effect = BrokenPipeError()
                self.stderr = mock.Mock()
                self.stderr.read.return_value = b"Pipe cerrado"
                self.returncode = None

            def wait(self):
                captured["waited"] = True
                self.returncode = 1

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(render_module.subprocess, "Popen", BrokenPipeProc):
                        with self.assertRaises(SystemExit) as ctx:
                            self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("Pipe cerrado", str(ctx.exception))
            self.assertTrue(captured["waited"])
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_export_os_replace_failure_preserves_destination_and_cleans_temp(self):
        n = 10
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                        with mock.patch.object(render_module.os, "replace", side_effect=OSError("bloqueado")):
                            with self.assertRaises(OSError) as ctx:
                                self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("bloqueado", str(ctx.exception))
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_atomic_output_unique_sibling_and_cleanup_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "video.mp4"
            first = None
            with render_module._atomic_output(dest) as temp:
                first = temp
                self.assertEqual(temp.suffix, ".mp4")
                self.assertEqual(temp.parent, dest.parent)
                self.assertTrue(temp.name.startswith(dest.stem + "."))
                temp.write_bytes(b"x")
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"x")
            self.assertFalse(first.exists())
            # A second publication uses a different unique temp.
            with render_module._atomic_output(dest) as second:
                self.assertNotEqual(second, first)
                second.write_bytes(b"y")
            self.assertEqual(dest.read_bytes(), b"y")
            # An exception inside the block cleans the temp and never touches dest.
            dest.write_bytes(b"final")
            with self.assertRaises(RuntimeError):
                with render_module._atomic_output(dest) as temp3:
                    temp3.write_bytes(b"z")
                    raise RuntimeError("abort")
            self.assertEqual(dest.read_bytes(), b"final")
            self.assertFalse(temp3.exists())

    def _debug(self, analysis, info, out, fake_proc):
        with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
            with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(int(len(analysis["keep"])))):
                with mock.patch.object(render_module, "_estimate_jpeg_bytes", return_value=0.0):
                    with mock.patch.object(render_module.subprocess, "Popen", fake_proc):
                        with contextlib.redirect_stdout(io.StringIO()):
                            stabilizer.write_debug(Path("v.mp4"), info, analysis, out, debug_width=320, max_images=10000)

    def test_debug_review_success_replaces_previous_and_leaves_no_temp(self):
        n = 4
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            review = out / "debug" / "review.mp4"
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_bytes(b"review previo")
            self._debug(self._analysis(n), self._info(n), out, self._fake_encoder(captured))
            self.assertTrue(review.exists())
            self.assertEqual(review.read_bytes(), b"temporal")
            self.assertEqual(self._temps(review.parent, "review"), [])

    def test_debug_review_failure_preserves_previous_and_cleans_temp(self):
        n = 4
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            review = out / "debug" / "review.mp4"
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_bytes(b"review previo")
            fake = self._fake_encoder(captured, returncode=1, stderr=b"review fallo")
            with self.assertRaises(SystemExit) as ctx:
                self._debug(self._analysis(n), self._info(n), out, fake)
            self.assertIn("review fallo", str(ctx.exception))
            self.assertEqual(review.read_bytes(), b"review previo")
            self.assertEqual(self._temps(review.parent, "review"), [])

    def test_export_os_replace_failure_prints_no_success_message(self):
        n = 10
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            stdout = io.StringIO()
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                        with mock.patch.object(render_module.os, "replace", side_effect=OSError("bloqueado")):
                            with contextlib.redirect_stdout(stdout):
                                with self.assertRaises(OSError) as ctx:
                                    self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("bloqueado", str(ctx.exception))
            # The success announcement only happens after os.replace: never here.
            self.assertNotIn("Vídeo escrito", stdout.getvalue())
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_export_stdin_close_oserror_reaps_and_cleans(self):
        n = 10
        captured = {}

        class CloseErrorProc:
            def __init__(self, command, **kwargs):
                captured["command"] = command
                Path(command[-1]).write_bytes(b"temporal")
                self.stdin = mock.Mock()
                self.stdin.close.side_effect = OSError("canal cerrado")
                self.stderr = mock.Mock()
                self.stderr.read.return_value = b""
                self.returncode = None
                self._final = 0

            def wait(self):
                captured["waited"] = True
                self.returncode = self._final

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=self._iter_frames(n)):
                    with mock.patch.object(render_module.subprocess, "Popen", CloseErrorProc):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(OSError) as ctx:
                                self._export(self._analysis(n), self._info(n), dest)
            self.assertIn("canal cerrado", str(ctx.exception))
            # The process is still reaped and the temporary cleaned despite the
            # non-BrokenPipe OSError on stdin.close.
            self.assertTrue(captured["waited"])
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])

    def test_export_decoder_runtime_error_preserves_destination_no_traceback(self):
        n = 10
        captured = {}

        def failing_frames(video, width, height, frames, *, exact_total=False):
            raise RuntimeError("fallo del decodificador")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out" / "stabilized.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"destino previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=failing_frames):
                    with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                self._export(self._analysis(n), self._info(n), dest)
            # A clean SystemExit, never a raw RuntimeError traceback.
            self.assertIn("fallo del decodificador", str(ctx.exception))
            self.assertFalse(isinstance(ctx.exception, RuntimeError))
            self.assertEqual(dest.read_bytes(), b"destino previo")
            self.assertEqual(self._temps(dest.parent, dest.stem), [])
            self.assertTrue(captured["waited"])

    def test_debug_decoder_runtime_error_preserves_review_no_traceback(self):
        n = 4
        captured = {}

        def failing_frames(video, width, height, frames, *, exact_total=False):
            raise RuntimeError("fallo del decodificador")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            review = out / "debug" / "review.mp4"
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_bytes(b"review previo")
            with mock.patch.object(render_module, "resolve_ffmpeg", return_value="ffmpeg"):
                with mock.patch.object(render_module, "iter_ffmpeg_bgr", side_effect=failing_frames):
                    with mock.patch.object(render_module, "_estimate_jpeg_bytes", return_value=0.0):
                        with mock.patch.object(render_module.subprocess, "Popen", self._fake_encoder(captured)):
                            with contextlib.redirect_stdout(io.StringIO()):
                                with self.assertRaises(SystemExit) as ctx:
                                    stabilizer.write_debug(
                                        Path("v.mp4"), self._info(n), self._analysis(n), out,
                                        debug_width=320, max_images=10000,
                                    )
            self.assertIn("fallo del decodificador", str(ctx.exception))
            self.assertFalse(isinstance(ctx.exception, RuntimeError))
            self.assertEqual(review.read_bytes(), b"review previo")
            self.assertEqual(self._temps(review.parent, "review"), [])
            self.assertTrue(captured["waited"])


if __name__ == "__main__":
    unittest.main()
