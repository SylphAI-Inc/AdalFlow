AdalFlow JSON Viewer
=============================

This is a simple viewer to display JSON data within an iframe. It helps in visualizing JSON data directly in the documentation.

.. raw:: html

    <style>
    .iframe-container {
        width: 100%;
        height: 100vh; /* Full height of the viewport */
        max-height: 1000px; /* Maximum height to ensure it doesn't get too tall on larger screens */
        overflow: hidden;
        position: relative;
    }
    .iframe-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    </style>

    <div class="iframe-container">
        <iframe src="../_static/files/constrained_max_steps_12_a1754_run_1.json"></iframe>
    </div>
