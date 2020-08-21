import cv2
import os

for root, dirs, files in os.walk('/home/wintermute/breast_cancer_analyzer_LCAD/dataset/new-dataset/Calc-Test'):
	f_spl = {}

	mrg = lambda p, k: p + 'CC_' + k
	
	for i, f in enumerate(files):
		pacient, kind = f.split('CC_')
		if pacient in f_spl: f_spl[pacient].append(kind)
		else: f_spl[pacient] = [kind]		
	
	for pacient, kind in f_spl.items():
		kind.sort(reverse=True)
		print(pacient)
		img = cv2.imread(root + '/' + mrg(pacient, kind[0]))
		left_half = img[:, :img.shape[1]/2, :]
		right_half = img[:, img.shape[1]/2:, :]
		if ((left_half == 0).sum() > (right_half == 0).sum()):
			print("Flipping")
			for k in kind:
				fname = root + '/' + mrg(pacient, k)
				img_flip = cv2.imread(fname)
				img_flip = cv2.flip(img_flip, 1)
				cv2.imwrite(fname, img_flip)
				del img_flip

		del img

