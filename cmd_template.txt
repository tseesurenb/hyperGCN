python main_2.py --layers=2 --u_sim_top_k=40 --i_sim_top_k=15 --u_sim=cosine --i_sim=cosine --edge=knn --model=LightGCNAttn --weight_mode=exp --epochs=51 --dataset=yelp2018 --verbose=1  

main.py --layers=1 --u_sim_top_k=35 --i_sim_top_k=10 --u_sim=cosine --i_sim=cosine --edge=knn --model=LightGCNAttn --weight_mode=exp --epochs=201 --verbose=1 --dataset=epinion --layers=1