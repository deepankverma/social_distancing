def mid_point (startX, startY, endX, endY, F):
	
	
	# Mid point of bounding box
	x_mid = round((startX+endX)/2,4)
	# y_mid = round((startY+endY)/2,4)

	### if taking the mid low point

	y_mid = round(endY,4)          

	#############    

	height = round(endY-startY,4)

	# Distance from camera based on triangle similarity
	distance = (165 * F)/height
	# print("Distance(cm):{dist}\n".format(dist=distance))

	# Mid-point of bounding boxes (in cm) based on triangle similarity technique
	x_mid_cm = (x_mid * distance) / F
	y_mid_cm = (y_mid * distance) / F
	pos_dict = (x_mid_cm,y_mid_cm,distance)

	return pos_dict, x_mid, y_mid

	##### transform the mid point ##########

	# print(M.shape)

	# points = (x_mid_cm, y_mid_cm)

	# print(points)
	
	# # ones = np.ones(shape=(len(points), 1))

	# # print(ones)

	# # points_ones = np.hstack([points, ones])

	# # print(points_ones.shape)
	# p = points
	# matrix = M
	# px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	# py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	# p_after = (int(px), int(py))


	# print("point_transform ====================", p_after)

	# htspts = cv2.circle(warped, p_after, 15, (255, 0, 0), 3)

	# cv2.imshow('Hotspots', htspts)