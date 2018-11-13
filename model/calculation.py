import numpy as np

def dice_coefficient(pred_label, label):

    pred_RVb = np.sum(pred_label == 1)
    pred_LVw = np.sum(pred_label == 2)
    pred_LVb = np.sum(pred_label == 3)

    true_RVb = np.sum(label == 1)
    true_LVw = np.sum(label == 2)
    true_LVb = np.sum(label == 3)

    pred_transform = pred_label.copy()
    pred_transform[pred_transform==0] = 255

    reference = true_RVb + true_LVw + true_LVb
    prediction = pred_RVb + pred_LVw + pred_LVb
    intersection = np.sum(pred_transform==label)

    if reference == 0 and prediction == 0:
        return 1
    else:
        dice = (2.0 * intersection) / ( reference + prediction )
        #if dice < 0.3:
            #print ("too tiny:", reference)
        return dice

def multi_dice_coefficient(pred_label, label):
    referencies = []
    predictions = []
    intersections = []

    for cross_section in range(label.shape[0]):
        pred_RVb = np.sum(pred_label[cross_section] == 1)
        pred_LVw = np.sum(pred_label[cross_section] == 2)
        pred_LVb = np.sum(pred_label[cross_section] == 3)

        true_RVb = np.sum(label[cross_section] == 1)
        true_LVw = np.sum(label[cross_section] == 2)
        true_LVb = np.sum(label[cross_section] == 3)

        pred_transform = pred_label[cross_section].copy()
        pred_transform[pred_transform==0] = 255

        reference = true_RVb + true_LVw + true_LVb
        prediction = pred_RVb + pred_LVw + pred_LVb
        intersection = np.sum(pred_transform==label[cross_section])

        referencies.append(reference)
        predictions.append(prediction)
        intersections.append(intersection)

    reference = sum(referencies)
    prediction = sum(predictions)
    intersection = sum(intersections)

    if reference == 0 and prediction == 0:
        return 1
    else:
        dice = (2.0 * intersection) / ( reference + prediction )
        return dice

def multi_dice_coefficient_classify(pred_label, label, target):
    referencies = []
    predictions = []
    intersections = []

    for cross_section in range(label.shape[0]):

        pred_target = np.sum(pred_label[cross_section] == target)
        true_target = np.sum(label[cross_section] == target)
        intersection = np.sum((pred_label[cross_section] == target) & (label[cross_section] == target))

        referencies.append(true_target)
        predictions.append(pred_target)
        intersections.append(intersection)

    reference = sum(referencies)
    prediction = sum(predictions)
    intersection = sum(intersections)

    if reference == 0 and prediction == 0:
        return 1
    else:
        dice = (2.0 * intersection) / ( reference + prediction )
        return dice

def compute_EF_gt(subject_number, labels_list):

    ED = labels_list[subject_number*2]
    ES = labels_list[subject_number*2 + 1]
    ED = np.transpose(ED, (2, 0, 1))
    ES = np.transpose(ES, (2, 0, 1))

    RV_EDV = compute_ventricle(ED, 'R')
    RV_ESV = compute_ventricle(ES, 'R')

    LV_EDV = compute_ventricle(ED, 'L')
    LV_ESV = compute_ventricle(ES, 'L')

    REF = ejection_fraction(list([RV_EDV,RV_ESV]))
    LEF = ejection_fraction(list([LV_EDV,LV_ESV]))

    return REF, LEF

def compute_LVb(input):
    Total_LVb = []

    for slice in range(input.shape[0]):
        LVb = np.sum(input[slice] == 3)
        Total_LVb.append(LVb)

    return sum(Total_LVb)

def compute_ventricle(input, side):
    Total_volumes = []

    for slice in range(input.shape[0]):

        if side == 'R' or side == 'r':
            volume = np.sum(input[slice] == 1)

        elif side == 'L' or side == 'l':
            volume = np.sum(input[slice] == 3)

        else:
            print ("Wrong input side, please check")
            return None

        Total_volumes.append(volume)

    return sum(Total_volumes)

def ejection_fraction(volumes_list):
    #conjecture_LVb = []
    #authentic_LVb = []
    SV = max(volumes_list) - min(volumes_list)
    EF = SV / max(volumes_list)
    return EF
