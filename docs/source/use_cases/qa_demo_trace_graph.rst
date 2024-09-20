Q&A Few Shot Demo Trace Graph
=============================

This demonstrates the Few-shot demostration trace graph using the `fit` method of the `Trainer` class when `debug` is set to `True`.
Compared with the computation graph, this trace graph traces and the proposed new value for the demo parameters.

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
    .zoom-controls {
        position: absolute;
        top: 10px;
        right: 10px;
        display: flex;
        gap: 10px;
    }
    .zoom-controls button {
        padding: 5px 10px;
        cursor: pointer;
    }
    </style>
    <div class="iframe-container">
        <iframe srcdoc="
            <html>
            <body style='margin:0; padding:0;'>
                <img id='zoomImage' src='../_static/images/trace_graph_EvalFnToTextLoss_output_id_6ea5da3c-d414-4aae-8462-75dd1e09abab.png' style='width:100%; height:auto; transform-origin: center center; transition: transform 0.25s ease;'>
            </body>
            </html>
        "></iframe>
    </div>
