cd src
nohup python one_pass.py fft one_pass 1500 1500 &
nohup python one_pass.py fft iterative_training 1500 1500 &
nohup python one_pass.py fft MCMA 1500 1500 &
nohup python one_pass.py fft MCMA_complementary 1500 1500 &

nohup python one_pass.py bessel_Jnu one_pass 1500 1500 &
nohup python one_pass.py bessel_Jnu iterative_training 1500 1500 &
nohup python one_pass.py bessel_Jnu MCMA 1500 1500 &
nohup python one_pass.py bessel_Jnu MCMA_complementary 1500 1500 &

nohup python one_pass.py blackscholes one_pass 1500 1500 &
nohup python one_pass.py blackscholes iterative_training 1500 1500 &
nohup python one_pass.py blackscholes MCMA 1500 1500 &
nohup python one_pass.py blackscholes MCMA_complementary 1500 1500 &

nohup python one_pass.py jmeint one_pass 1500 1500 &
nohup python one_pass.py jmeint iterative_training 1500 1500 &
nohup python one_pass.py jmeint MCMA 1500 1500 &
nohup python one_pass.py jmeint MCMA_complementary 1500 1500 &

nohup python one_pass.py jpeg one_pass 1500 1500 &
nohup python one_pass.py jpeg iterative_training 1500 1500 &
nohup python one_pass.py jpeg MCMA 1500 1500 &
nohup python one_pass.py jpeg MCMA_complementary 1500 1500 &

nohup python one_pass.py inversek2j one_pass 1500 1500 &
nohup python one_pass.py inversek2j iterative_training 1500 1500 &
nohup python one_pass.py inversek2j MCMA 1500 1500 &
nohup python one_pass.py inversek2j MCMA_complementary 1500 1500 &

nohup python one_pass.py sobel one_pass 1500 1500 &
nohup python one_pass.py sobel iterative_training 1500 1500 &
nohup python one_pass.py sobel MCMA 1500 1500 &
nohup python one_pass.py sobel MCMA_complementary 1500 1500 &

nohup python one_pass.py kmeans one_pass 1500 1500 &
nohup python one_pass.py kmeans iterative_training 1500 1500 &
nohup python one_pass.py kmeans MCMA 1500 1500 &
nohup python one_pass.py kmeans MCMA_complementary 1500 1500 &
