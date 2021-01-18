# kubectl delete -f solo-cloud-pod.yaml
kubectl delete -f edge-pod.yaml
# kubectl delete -f cloud-pod.yaml

kubectl delete -f kafka-pod-3.yaml
kubectl delete -f kafka-pod-2.yaml
kubectl delete -f kafka-pod.yaml
kubectl delete -f kafka-service.yaml
kubectl delete -f kafka-service-2.yaml
kubectl delete -f kafka-service-3.yaml

kubectl delete -f zookeeper-pod.yaml
kubectl delete -f zookeeper-service.yaml
