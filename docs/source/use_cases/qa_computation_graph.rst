Q&A Computation Graph
=============================

This demonstrates the computation graph by using the output parameter of a task pipeline with method `draw_graph`.

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
                <img id='zoomImage' src='../_static/images/trace_graph_Generator_output_id_689cc5a1-6737-40a8-8faa-8bbf7bddfed8.png' style='width:100%; height:auto; transform-origin: center center; transition: transform 0.25s ease;'>
            </body>
            </html>
        "></iframe>
    </div>
