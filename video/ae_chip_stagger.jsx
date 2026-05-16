// ae_chip_stagger.jsx
// Run from After Effects via File > Scripts > Run Script File...
//
// What it does:
//   Takes the currently active comp and applies a staggered fade-in to every
//   selected layer. Defaults match animation_spec.json (1.4 s spacing, 0.25 s
//   fade, easeOutCubic). Adjust constants below if you need different timing.
//
// How to use:
//   1. Open your comp in AE.
//   2. Select the FAIL chip layers in the order you want them to appear
//      (top-to-bottom, left-to-right — AE selection order is preserved).
//   3. File > Scripts > Run Script File... and pick this file.

(function () {
    var STAGGER_SECONDS  = 1.4;   // Gap between successive chips.
    var FADE_SECONDS     = 0.25;  // Length of each fade-in.
    var START_OFFSET     = 1.5;   // Seconds after current time indicator to begin.
    var SCALE_FROM_PCT   = 96;    // Starting scale percent.
    var SCALE_TO_PCT     = 100;   // Final scale percent.

    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {
        alert("ae_chip_stagger.jsx — no active comp.");
        return;
    }
    var layers = comp.selectedLayers;
    if (!layers || layers.length === 0) {
        alert("ae_chip_stagger.jsx — select at least one layer first.");
        return;
    }

    app.beginUndoGroup("Chip stagger");

    var t0 = comp.time + START_OFFSET;

    for (var i = 0; i < layers.length; i++) {
        var layer = layers[i];
        var inTime = t0 + i * STAGGER_SECONDS;
        var outTime = inTime + FADE_SECONDS;

        // Opacity 0 → 100
        var opacity = layer.property("ADBE Transform Group").property("ADBE Opacity");
        opacity.setValueAtTime(inTime, 0);
        opacity.setValueAtTime(outTime, 85);  // 85% to match design system

        // Scale SCALE_FROM_PCT → SCALE_TO_PCT
        var scale = layer.property("ADBE Transform Group").property("ADBE Scale");
        var cur = scale.value;
        scale.setValueAtTime(inTime, [SCALE_FROM_PCT, SCALE_FROM_PCT]);
        scale.setValueAtTime(outTime, [SCALE_TO_PCT, SCALE_TO_PCT]);

        // Set easing on the second keyframe (easeOutCubic ≈ ease out 66.7%)
        var ease = new KeyframeEase(0, 66.7);
        scale.setTemporalEaseAtKey(2, [ease, ease], [ease, ease]);
        opacity.setTemporalEaseAtKey(2, [ease], [ease]);
    }

    app.endUndoGroup();
    alert("ae_chip_stagger.jsx — applied to " + layers.length + " layers.");
})();
