"""Functions from G2D student.py, MIT licensed."""
import torch.nn as nn

def distillation_loss(student_logits, teacher_logits, temp):
    """
    Compute the distillation loss between the student's and teacher's logits.
    """
    soft_teacher_logits = nn.functional.softmax(teacher_logits / temp, dim=1)
    soft_student_logits = nn.functional.log_softmax(student_logits / temp, dim=1)
    return nn.functional.kl_div(soft_student_logits, soft_teacher_logits, reduction='batchmean') * (temp ** 2)


def feature_loss(student_features, teacher_features):
    """
    Compute the feature loss between the student's and teacher's features.
    """
    return nn.functional.mse_loss(student_features, teacher_features)


