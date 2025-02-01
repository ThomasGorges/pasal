python3 main.py --study_name ablation_1_dataset_embeddings_seed_0 --num_workers 5 --num_models 100 --use_embeddings --exclude arctander,ifra_2019,leffingwell,sigma_2014 --fold_id 999 --random_search_seed 0
python3 main.py --study_name ablation_2_dataset_embeddings_seed_0 --num_workers 5 --num_models 100 --use_embeddings --exclude arctander,ifra_2019,leffingwell --fold_id 999 --random_search_seed 0
python3 main.py --study_name ablation_3_dataset_embeddings_seed_0 --num_workers 5 --num_models 100 --use_embeddings --exclude arctander,leffingwell --fold_id 999 --random_search_seed 0
python3 main.py --study_name ablation_4_dataset_embeddings_seed_0 --num_workers 5 --num_models 100 --use_embeddings --exclude arctander --fold_id 999 --random_search_seed 0
python3 main.py --study_name ablation_5_dataset_embeddings_seed_0 --num_workers 5 --num_models 100 --use_embeddings --fold_id 999 --random_search_seed 0

python3 main.py --study_name ablation_1_dataset_embeddings_seed_0 --fetch_results --fold_id 999
python3 main.py --study_name ablation_2_dataset_embeddings_seed_0 --fetch_results --fold_id 999
python3 main.py --study_name ablation_3_dataset_embeddings_seed_0 --fetch_results --fold_id 999
python3 main.py --study_name ablation_4_dataset_embeddings_seed_0 --fetch_results --fold_id 999
python3 main.py --study_name ablation_5_dataset_embeddings_seed_0 --fetch_results --fold_id 999
