# -----------------------------------------------------------------------
#  Monitoring classes
# -----------------------------------------------------------------------
class ActivationMonitor:
    """
    A class for monitoring activations in PyTorch neural network modules.

    This class uses forward hooks to capture input and output tensors of specified layers.
    It performs a depth-first search through the model's modules using `named_modules()`.

    Key behaviors:
    - Monitors all layers if no specific layers are specified.
    - Captures both pre-activation (input) and post-activation (output) tensors.
    - Includes the root module (entire network) with an empty string key ('').

    The activations dictionary keys correspond to module names:
    - 'layer_name': For specific layers
    - 'container_name': For container modules like nn.Sequential, nn.ModuleList, nn.ModuleDict
    - '': For the root module (entire network)

    Each activation entry contains:
    - 'pre': Input tensor to the module
    - 'pos': Output tensor from the module

    Note: The root module ('') provides the overall network input and final output.

    Args:
        model (nn.Module): The PyTorch model to monitor.
        modules_to_monitor (list, optional): Specific modules to monitor. If None, all modules are monitored.

    Example:
        monitor = ActivationMonitor(model)
        output = model(input_data)
        activations = monitor.activations
    """
    def __init__(self, model, modules_to_monitor = None):
        self.model              = model
        self.modules_to_monitor = modules_to_monitor
        self.activations        = {}
        self.hooks              = []

    def hook_fn(self, name):
        """ Closure to parameterize a function: f(params)(input) """
        def hook(module, input, output):
            input_tensor  = input[0].detach()  # For some reasons, it's always a tuple
            output_tensor = output.detach()
            self.activations[name] = {
                'pre' : (input_tensor.mean().item() , input_tensor.std().item()),
                'pos' : (output_tensor.mean().item(), output_tensor.std().item()),
            }
        return hook

    def add_hooks(self):
        for name, module in self.model.named_modules():
            if self.modules_to_monitor is None or isinstance(module, self.modules_to_monitor):
                getattr(module, 'remove_hooks', lambda : None)()
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)  # For clean-up

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradientMonitor:
    """
    A class for monitoring gradients in PyTorch neural network parameters.

    This class uses backward hooks to capture gradient tensors of specified parameters
    during the backward pass.  It iterates through the model's parameters using `named_parameters()`.

    Key behaviors:
    - Monitors gradients of all parameters if no specific layers are specified.
    - Uses parameter hooks to capture gradients after they've been computed.
    - Stores gradients for each monitored parameter.

    The gradients dictionary keys correspond to parameter names:
    - 'layer_name.weight': For weight parameters
    - 'layer_name.bias': For bias parameters

    Note: Unlike ActivationMonitor, there's no root module concept here. Only individual
    parameter gradients are captured.

    Args:
        model (nn.Module): The PyTorch model to monitor.
        params_to_monitor (list, optional): Specific parameters to monitor. If None, all parameters are monitored.
                                            Partial matches are allowed (e.g., 'conv' will match 'conv1', 'conv2', etc.)

    Attributes:
        gradients (dict): A dictionary storing the gradients of monitored parameters.
                          Keys are parameter names, values are gradient tensors.

    Example:
        monitor = GradientMonitor(model.vit.encoder)  # You can pass in the entire model or just a sub-module
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        gradients = monitor.gradients

    Note:
    - Gradients are cloned and moved to CPU to avoid interfering with the original computation.
    - Hooks should be removed (using remove_hooks()) when no longer needed to free up memory.
    """
    def __init__(self, model, params_to_monitor = None):
        self.model             = model
        self.params_to_monitor = set(params_to_monitor) if params_to_monitor is not None else None
        self.gradients         = {}
        self.hooks             = []

    def hook_fn(self, name):
        def hook(grad):
            """ https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook """
            self.gradients[name] = grad.clone().cpu()
        return hook

    def add_hooks(self):
        for name, param in self.model.named_parameters():
            if self.params_to_monitor is None or any(layer in name for layer in self.params_to_monitor):
                getattr(param, 'remove_hooks', lambda : None)()
                hook = param.register_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ----------------------------------------------------------------------- #
#  Helper functions
# ----------------------------------------------------------------------- #
def get_param(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param
    raise ValueError(f"No parameter named '{param_name}' found in the model.")


def get_param_grad(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param.grad
    raise ValueError(f"No parameter named '{param_name}' found in the model.")


def monitor_param_update_metrics(model, lr, params_to_monitor = None):
    """
    Monitor and compute metrics for specified parameters in the model.

    This function calculates two metrics for each monitored parameter:
    - Percent parameter update: Order of magnitude of the ratio between the
      standard deviation of the parameter update and the standard deviation of
      the parameter itself.
    - Gradient mean and standard deviation.

    Args:
        model (nn.Module): The PyTorch model to monitor.
        lr (float): The learning rate used in optimization.
        params_to_monitor (list, optional): Specific parameters to monitor. 
            If None, all parameters with gradients are monitored.

    Note:
        Users need to specify full parameter names, including '.weight' or '.bias'.

    Example:
        monitor_param_metrics(model, lr, ['features.0.weight', 'features.0.bias', 'features.2.weight'])

    Returns:
        dict: A dictionary containing metrics for each monitored parameter.
              Keys are 'percent_param_update' and 'grad_mean_std'.
    """
    metrics = {
        'percent_param_update': {},
        'grad_mean_std': {}
    }

    for name, param in model.named_parameters():
        if params_to_monitor is not None and name not in params_to_monitor:
            continue

        if param.grad is not None:
            # Percent param update
            metrics['percent_param_update'][name] = (
                (param.grad * lr).std().cpu() /
                param.detach().std().cpu()
            ).log10().item()

            # Mean and Std of gradients
            metrics['grad_mean_std'][name] = (
                param.grad.mean().cpu().item(),
                param.grad.std().cpu().item()
            )

    return metrics


def create_param_update_monitor(model, lr, weights_only = True):
    """
    Create a closure (currying) for monitoring parameter update metrics.

    This function calculates two metrics for each monitored parameter:
    - Percent parameter update: Order of magnitude of the ratio between the
      standard deviation of the parameter update and the standard deviation of
      the parameter itself.
    - Gradient mean and standard deviation.

    Args:
        model (nn.Module): The PyTorch model or sub-module to monitor.
        lr (float): The learning rate used in optimization.
        weights_only (bool, optional): Monitor weights only.  Defaults to True.

    Returns:
        function: A closure that when called, returns the monitored metrics.

    Note:
        Users need to specify full parameter names, including '.weight' or '.bias'.

    Example:
        monitor = create_param_update_monitor(model, lr)
        results = monitor()
    """
    def monitor():
        metrics = {
            'percent_param_update': {},
            'grad_mean_std': {}
        }
        for name, param in model.named_parameters():
            # Conditionally skip biases
            if weights_only and '.bias' in name:
                continue

            if param.grad is not None:
                # Percent param update
                metrics['percent_param_update'][name] = (
                    (param.grad * lr).std().cpu() /
                    param.detach().std().cpu()
                ).log10().item()

                # Mean and Std of gradients
                metrics['grad_mean_std'][name] = (
                    param.grad.mean().cpu().item(),
                    param.grad.std().cpu().item()
                )

        return metrics

    return monitor
