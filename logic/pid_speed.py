def PID_speed(self, setpoint, measured_value, previous_error):     # PID CONTROL FOR TURNS ONLY
	# PID Control VARIABLES
	error = setpoint - measured_value
	integral = 0.0
	# DYNAMIC PID CONTROL VARIABLES
	if error > 0.5:
	    Kp = 0.09 * 5
	    Ki = 0.00144 * 5 
	    Kd = 1.40625 * 5
	else:
	    Kp = 0.09 * 3
	    Ki = 0.00144 * 3
	    Kd = 1.40625 * 3
	integral += error
	derivative = error - previous_error
	previous_error = error
	return Kp * error + Ki * integral + Kd * derivative, error
