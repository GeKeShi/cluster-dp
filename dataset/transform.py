input_file = raw_input("enter the input file name:")
output_file = raw_input("enter the result file name:")
location = []
# input_lable = []
for line in open(input_file, "r"):
    line = line.replace('-','')
    items = line.strip("\n").split(":")
    # input_lable.append(int(items.pop()))
    tmp = []
    tmp.append(float(items[0]))
    tmp.append(float(items[1]))
    location.append(tmp)

resultfile=open(output_file, "w")
location=[str(item[0]/10000)+","+str(item[1]/10000)+'\n'for item in location]
	# print str(item[0])+","+str(item[1])
resultfile.writelines(location)
resultfile.close()