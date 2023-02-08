from common.utils.box_ops import box_scale, box_clip


def detector_postprocess(results, output_height: int, output_width: int):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (dict): the raw outputs from the detector.
            `results['image_size']` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height: the desired output resolution.
        output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    input_height, input_width = results['image_size'][0], results['image_size'][1]
    new_size = (output_height, output_width)
    output_width_tmp = output_width
    output_height_tmp = output_height

    scale_x, scale_y = (output_width_tmp / input_width, output_height_tmp / input_height,)

    if 'pred_boxes' in results:
        results['pred_boxes'] = box_scale(results['pred_boxes'], scale_x, scale_y)
        results['pred_boxes'] = box_clip(results['pred_boxes'], clip_size=new_size)

    return results

