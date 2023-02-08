import mindspore as ms

ms.ops.distribute
ms.communication.GlobalComm.INITED

def is_dist_avail_and_initialized() -> bool:
    """
    Checking if the distributed package is available and
    the default process group has been initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True