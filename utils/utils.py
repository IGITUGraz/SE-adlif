
import numpy as np

def save_fig_to_aim(logger, figure, name, epoch_step):
    try: 
        from aim import Image
        
        fig = Image(figure, optimize=True)
        logger.experiment.track(
            fig,
            name=name,
            step=epoch_step,
        )
    except ImportError:
        pass

def save_distributions_to_aim(logger, distributions, name, epoch_step):
    try:
        from aim import Distribution    
        for k, v in distributions:
            dist = Distribution(v)
            logger.experiment.track(
                dist,
                name=f"{name}_{k}",
                step=epoch_step,
            )
    except ImportError:
        pass
    
def get_event_indices(data):
    num_dimensions = data.shape[0]
    event_indices = [np.where(data[dim] == 1)[0] for dim in range(num_dimensions)]
    return event_indices