microk8s kubectl delete pod amir-pod-project
microk8s kubectl apply -f pipeline-parallelism.yaml
microk8s kubectl logs -f amir-pod-project