Q&A Text Grad Trace Graph
=============================

This demonstrates the Text Grad trace graph using the `fit` method of the `Trainer` class when `debug` is set to `True`.
Compared with the computation graph, this trace graph traces (1) batch-wise gradients, (2) the loss, (3) and the proposed new value for the prompt parameters.
This example in particular shows a batch size of 2.

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
                <img id='zoomImage' src='../_static/images/trace_graph_sum_id_e53cb8f9-235d-480b-b630-f480a9dfb5d0.png' style='width:100%; height:auto; transform-origin: center center; transition: transform 0.25s ease;'>
            </body>
            </html>
        "></iframe>
    </div>
