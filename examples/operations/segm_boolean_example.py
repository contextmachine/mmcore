import itertools
import json
import numpy as np

from mmcore.topo.curve_boolean import segment_boolean
from pathlib import Path
path=Path(__file__).parent/'segm_boolean_test_data.json'
with open(path, "r") as f:
    data = json.load(f)


regions=segment_boolean(data,tol=0.05)
print(f'{len(regions)} regions have been extracted:')
for i,region in enumerate(regions):
    print(f'Region {i} containing {len(region)} segments.')
try:

    from mmcore.renderer.renderer2d import Renderer2D,RenderColorsConfig
    # Initialize Renderer
    renderer = Renderer2D()
    initial = np.array(data)


    # Add intersection markers

    # renderer.add_marker(intersection_points, color=args.marker_color, size=args.marker_size)

    # Prepare objects to render
    objects_to_render =[np.array(i) for i in itertools.chain.from_iterable(regions)]


    # Render the scene
    rendered_image=renderer( objects_to_render+list(initial) , colors=((['orange']*len(objects_to_render))+['gray']*len(initial)),linewidth=(([2]*len(objects_to_render))+[1]*len(initial)))

    # Save or display the image

    rendered_image.write_image(path.parent/'segm_boolean_result.png')
    rendered_image.show()

except ModuleNotFoundError:
    print('pass')
