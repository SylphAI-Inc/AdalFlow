AdalFlow Trace Graph
=============================

AdalFlow Trace Graph is able to visualize the DAG of the task pipeline and its training gradients along with the proposed value.
This visualization is especially helpful for debugging.

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
                <img id='zoomImage' src='../_static/images/trace_graph_sum.png' style='width:100%; height:auto; transform-origin: center center; transition: transform 0.25s ease;'>
            </body>
            </html>
        "></iframe>
    </div>
