import numpy as np

waypoints_dict = {
    'neutral': np.array([0.495,0.0339,0.312,0.878,-0.473,0.063,-0.028]),
    'relative_plate': np.array([-0.008, -0.042, 0.031,0.633, 0.774, 0.012, 0.028]),
    'toaster_waypoint': np.array([0.5858,-0.28,0.3,0.9735,-0.198,0.091,-0.116]),
    'toaster_absolute': np.array([0.593,-0.311,0.1826,0.716,-0.694,0.066,-0.007]),
    'switch_pre': np.array([0.3168,-0.517,0.385,0.81,-0.254,0.526,-0.04045]),
    'switch_on_wp': np.array([0.47,-0.52, 0.214,0.696,-0.127,0.706,-0.0376]),
    'switch_on': np.array([0.471,-0.52,0.13,0.696,-0.127,0.7058,-0.0377]),
    'switch_off_wp': np.array([0.46751,-0.49899,0.1937,0.687,-0.0702,0.717,-0.091]),
    'switch_off': np.array([0.54457,-0.4998,0.19651,0.68843,-0.049,0.71553,-0.108])
}