import const


with open(const.dataset_dir + "/民法典.txt","r") as f:
	txt = f.read().split("\n");

for l in txt:
	print(l.strip());
