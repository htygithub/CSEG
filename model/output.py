import os
import numpy as np

from scipy.misc import imsave

def overlapping(image, label, output_name, folder):
    #Set label RGB
    label_c = Color_Label(label)

    #image filted
    image_filted = image.copy()
    image_filted[label>0] = 0

    image_c = np.dstack([image_filted.astype(np.uint8)] * 3).copy(order='C')

    overlap = image_c + label_c

    imsave(os.path.join(folder, output_name), overlap)

def Color_Label(label):
    label_r = label.copy()
    label_g = label.copy()
    label_b = label.copy()
    label_r[label==1], label_g[label==1], label_b[label==1] = 55,140,220
    label_r[label==2], label_g[label==2], label_b[label==2] = 115,255,20
    label_r[label==3], label_g[label==3], label_b[label==3] = 255,255,0

    color_label = np.dstack((label_r,label_g,label_b))

    return color_label

def csv_output(id_, ED_dice, ED_RVb_dice, ED_LVw_dice, ED_LVb_dice, ED_failure_rate, ES_dice, ES_RVb_dice, ES_LVw_dice, ES_LVb_dice, ES_failure_rate, REF, LEF, t_REF, t_LEF):
    import csv
    #global CSV_NAME
    CSV_NAME = 'Output.csv'
    title = ['Patient_ID',
             'ED_Dice', 'ED_RVb_Dice', 'ED_LVw_Dice', 'ED_LVb_Dice', 'ED_Failure_rate', 'ES_Dice', 'ES_RVb_Dice', 'ES_LVw_Dice', 'ES_LVb_Dice', 'ES_Failure_rate',
             'REF_vaule', 'LEF_vaule', 'True_REF_vaule', 'True_LEF_vaule']
    rows = zip(*[id_, ED_dice, ED_RVb_dice, ED_LVw_dice, ED_LVb_dice, ED_failure_rate, ES_dice, ES_RVb_dice, ES_LVw_dice, ES_LVb_dice, ES_failure_rate, REF, LEF, t_REF, t_LEF])

    with open(CSV_NAME, "w") as f:
        writer = csv.writer(f)
        writer.writerow(title)
        for row in rows:
            writer.writerow(row)
