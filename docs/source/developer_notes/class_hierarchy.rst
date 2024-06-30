Class Hierarchy
=============================
From the plot of the `LightRAG` library's class hierarchy, we can see the library is well-centered around two base classes: `Component` and `DataClass`, and it has no more than two levels of subclasses.
This design philosophy results in a library with bare minimum abstraction, providing developers with maximum customizability.

.. raw:: html

    <style>
    .iframe-container {
        width: 100%;
        height: 100vh; /* Full height of the viewport */
        max-height: 1000px; /* Maximum height to ensure it doesn't get too tall on larger screens */
        overflow: hidden;
    }
    .iframe-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    @media (max-width: 768px) {
        .iframe-container {
            height: 60vh; /* Adjust height for mobile viewports */
            max-height: none; /* Remove the maximum height constraint for small screens */
        }
    }
    </style>

    <div class="iframe-container">
        <iframe src="../_static/class_hierarchy.html"></iframe>
    </div>
