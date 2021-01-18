kubectl apply -f zookeeper-pod.yaml
kubectl apply -f zookeeper-service.yaml

kubectl apply -f kafka-pod.yaml
kubectl apply -f kafka-pod-2.yaml
kubectl apply -f kafka-pod-3.yaml

kubectl apply -f kafka-service.yaml
kubectl apply -f kafka-service-2.yaml
kubectl apply -f kafka-service-3.yaml

kubectl apply -f edge-pod.yaml
# kubectl apply -f cloud-pod.yaml
# kubectl apply -f solo-cloud-pod.yaml

# SOLO CLOUD - TEST
# python code/main.py --name DEVICE --cifar10 --tensorflow --producer-front 104.199.44.40:9094,104.155.56.234:9095,35.240.114.107:9096 --producer-front-topic devicecloud --consumer-front 104.199.44.40:9094,104.155.56.234:9095,35.240.114.107:9096 --consumer-front-topic clouddevice

# EDGE - TEST
# python code/main.py --name DEVICE --cifar10 --tensorflow --producer-front localhost:9094  --producer-front-topic deviceedge --consumer-front localhost:9094  --consumer-front-topic edgedevice
# EDGE - 3 BROKERS TEST
# python code/main.py --name DEVICE --cifar10 --tensorflow --producer-front "localhost:9094,localhost:9095,localhost:9096"  --producer-front-topic deviceedge --consumer-front "localhost:9094,localhost:9095,localhost:9096"  --consumer-front-topic edgedevice
