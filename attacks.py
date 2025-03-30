import torch
import torch.nn.functional as F
import torch

# Defining class that handles Adversarial Attack Implementation (FGSM, PGD)


class AdversarialAttacks:
    """Handle Attack Implementation and logging"""

    def fgsm_attack(self, images, model, epsilon):
        """Function handling Fast Gradient Sign Attack Implementation"""

        # Set requires_grad for the entire batch
        images = images.requires_grad_(True)
        output = model(images)  # Forward pass

        # Assuming you want to use the first output as a target
        # This may need adjustment based on your loss
        target = torch.zeros_like(output)
        loss = torch.nn.functional.mse_loss(output, target)

        model.zero_grad()  # Zero gradients from previous steps
        loss.backward()  # Compute gradients

        # Create the adversarial images using the gradient sign
        gradient_sign = images.grad.data.sign()
        adversarial_images = images + epsilon * gradient_sign

        # Clamp to ensure the values are valid image pixels (between 0 and 1)
        adversarial_images = torch.clamp(adversarial_images, 0, 1)

        return adversarial_images

    def pgd_attack(self, images, model, epsilon, alpha, num_iter):
        """Function handling PGD Attack implementation"""

        # Ensure the images require gradients
        images = images.requires_grad_(True)
        original_images = images.clone().detach()
        adversarial_images = images.clone().detach().requires_grad_(True)

        for _ in range(num_iter):
            output = model(adversarial_images)  # Forward pass
            # Using MSE loss as in your FGSM example; adjust target as needed.
            target = torch.zeros_like(output)
            loss = F.mse_loss(output, target)

            model.zero_grad()
            # Compute gradients using torch.autograd.grad() instead of loss.backward()
            grad = torch.autograd.grad(
                loss, adversarial_images, retain_graph=False, create_graph=False)[0]

            # Check if grad is None, which indicates something went wrong
            if grad is None:
                print("Warning: Gradient is None. Skipping update.")
                break

            # Update adversarial images using the sign of the gradient
            adversarial_images = adversarial_images + alpha * grad.sign()

            # Clip the perturbation to be within [-epsilon, epsilon]
            perturbation = torch.clamp(
                adversarial_images - original_images, min=-epsilon, max=epsilon)
            adversarial_images = torch.clamp(
                original_images + perturbation, min=0, max=1)

            # Re-enable gradient tracking for the next iteration
            adversarial_images = adversarial_images.detach().requires_grad_(True)

        return adversarial_images
