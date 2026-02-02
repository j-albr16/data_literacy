newspaper_comparison:
	python3 process.py \
		--start=0\
		--num_images=1000\
		--data_dir data\
		--politician_reference_csv=politicians/data.csv \
		--article_data_csv="politician_image_dataset/data_info.csv"\
		--out_name=results/politician_face_extracted_dataset.csv\
