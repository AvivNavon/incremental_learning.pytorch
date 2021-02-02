import torch
import numpy as np
import torch.nn.functional as F


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def calc_ece(logits, targets, temp):
    probs = F.softmax(logits / temp)

    confidences = probs.max(-1).values.detach().numpy()
    accuracies = probs.argmax(-1).eq(targets).numpy()

    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    plot_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            plot_acc.append(accuracy_in_bin)
        else:
            plot_acc.append(0.0)

    return ece


if __name__ == '__main__':

    target = torch.randint(high=10, size=(100, ))
    logit = F.softmax(torch.randn((100, 10)))


    print(calc_ece(logit, target, .1))
