read -r -p "It will recreate container. Are you sure? [y/N] " response
case "$response" in
[yY][eE][sS]|[yY]) 
docker stop ulstu-devel
docker rm ulstu-devel
./start.sh
;;
*)
;;
esac
