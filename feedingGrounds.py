from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import argparse
import json
#https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

#default command: python feedingGrounds.py --whole_day 1 --ml_type comp

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", default='data.json')
parser.add_argument("--add_data", required=False, help="Enter filename to a json file with a dictionary containing a key called times and a value that is an array with new times to add to the dataset. Will rewrite data.json to new dataset. Make sure json file is in same directory as feedingGrounds.py")
parser.add_argument("--val_time", required=False, type=int, help="Enter a time to predict feeding for. also use the --val_prec to add the number of preceeding feedings there were")
parser.add_argument("--val_prec", required=False, type=int, help="Enter the number of preceeding feedings there were")
parser.add_argument("--whole_day", default=0, type=int, help="Options: {0,1}. Enter '1' if you want to generate values for the whole day. Enter '0' for only the entered time. Default is '0.'")
parser.add_argument("--ml_type", default='lin', type=str, help="Options: {'lin', 'grad', 'comp'}. Enter 'grad' to run a gradient regression instead of a linear regression. 'lin' is the default. 'comp' compares the output of lin and grad and outputs the best value")
args = parser.parse_args()

f = open(args.training_data, "r")
final = json.load(f)
f.close()
f = open(args.training_data, "w+")

def addNewData(newData):
	global final
	final["timesFed"].append(newData['times'])
	timesFed = final['timesFed']
	times = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
	feeding=[]
	temp=[]
	for i in range(9):
		for j in range(len(timesFed)):
			temp.append(0)
		feeding.append(temp)
		temp=[]


	def count(time, i):
		feeding[time//100-9][i]+=1

	for i in range(len(timesFed)):
		for time in timesFed[i]:
			count(time, i)

	feeding2 = []
	for i in range(len(feeding)):
		for j in range(len(feeding[i])):
			feeding2.append(feeding[i][j])

	final = {"times": [], "before": feeding2, "feeding": feeding2, "2d": [], "timesFed": timesFed}

	for i in range(len(times)):
		for j in range(len(timesFed)):
			final["times"].append(times[i])

	for i in range(len(final["before"])-1, 2, -1):
		final['before'][i]=final['before'][i-3]
	final["before"][0]=0
	final["before"][1]=0
	final["before"][2]=0

	#NOTE: combine final['times'] and ['before'] into a two dimensional array
	for i in range(len(final['times'])):
		temp.append(final['times'][i])
		temp.append(final['before'][i])
		final["2d"].append(temp)
		temp = []

try:
	newData = json.load(open(args.add_data, "r"))
	addNewData(newData)
except TypeError:
	None
except FileNotFoundError:
	print("Adding new data did not work. File Not Found.")


# print(final["times"])
# print(final['feeding'])

# plt.scatter(final["times"], final["feeding"])

# plt.show()

json.dump(final, f)
f.close()

#Organize data splits
x_train , x_test , y_train , y_test = train_test_split(final["2d"], final["feeding"], test_size = 0.10)

#initialize linear regression and gradient objects
reg = LinearRegression()
clf = GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')

#Linear Prediction functions
def lin_predictWhole():
	global reg
	pre = [[900,0]]
	temp = []
	for i in range(1,9):
		temp.append(pre[i-1][0]+100)
		temp.append(reg.predict([pre[i-1]])[0])
		pre.append(temp)
		temp=[]
	for i in range(8):
		if pre[i][0]==900:
			print("Time: {}  || Value: {}".format(pre[i][0],int(round(pre[i+1][1]))))
		else:
			print("Time: {} || Value: {}".format(pre[i][0],int(round(pre[i+1][1]))))


def lin_predictOnce(time, prec):
	global reg
	print("Time: {} || Value: {}".format(time,int(round(reg.predict([[time, prec]])[0]))))

#Gradient Prediction Functions

def grad_predictWhole():
	global reg
	pre = [[900,0]]
	temp = []
	for i in range(1,9):
		temp.append(pre[i-1][0]+100)
		temp.append(clf.predict([pre[i-1]])[0])
		pre.append(temp)
		temp=[]
	for i in range(8):
		if pre[i][0]==900:
			print("Time: {}  || Value: {}".format(pre[i][0],int(round(pre[i+1][1]))))
		else:
			print("Time: {} || Value: {}".format(pre[i][0],int(round(pre[i+1][1]))))


def grad_predictOnce(time, prec):
	global reg
	print("Time: {} || Value: {}".format(time,int(round(clf.predict([[time, prec]])[0]))))


def main():
	if args.ml_type=='lin':
		if args.val_time!=None and args.val_prec!=None:
			if args.whole_day==0:
				reg.fit(x_train,y_train)
				print("Accuracy: {}".format(reg.score(x_test,y_test)))
				lin_predictOnce(args.val_time, args.val_prec)
		elif args.whole_day==1:
			reg.fit(x_train,y_train)
			print("Accuracy: {}".format(reg.score(x_test,y_test)))
			lin_predictWhole()
		else:
			print("Either args.val_time/args.val_prec is not defined OR whole_day is not defined! Define them using --val_time or --val_prec.")
	elif args.ml_type=='grad':
		if args.val_time!=None and args.val_prec!=None:
			if args.whole_day==0:
				clf.fit(x_train,y_train)
				print("Accuracy: {}".format(clf.score(x_test,y_test)))
				grad_predictOnce(args.val_time, args.val_prec)
		elif args.whole_day==1:
			clf.fit(x_train,y_train)
			print("Accuracy: {}".format(clf.score(x_test,y_test)))
			grad_predictWhole()
		else:
			print("Either args.val_time/args.val_prec is not defined OR whole_day is not defined! Define them using --val_time or --val_prec.")
	elif args.ml_type=='comp':
		if args.val_time!=None and args.val_prec!=None:
			if args.whole_day==0:
				reg.fit(x_train,y_train)
				clf.fit(x_train,y_train)
				regg=reg.score(x_test,y_test)
				clff=clf.score(x_test,y_test)
				if regg>clff:
					print("Running LinearRegression...")
					print("Accuracy: {}".format(regg))
					lin_predictOnce(args.val_time, args.val_prec)
				else: 
					print("Running GradientBoostingRegressor...")
					print("Accuracy: {}".format(clff))
					grad_predictOnce(args.val_time, args.val_prec)
		elif args.whole_day==1:
			reg.fit(x_train,y_train)
			clf.fit(x_train,y_train)
			regg=reg.score(x_test,y_test)
			clff=clf.score(x_test,y_test)
			if regg>clff:
				print("Running LinearRegression...")
				print("Accuracy: {}".format(regg))
				lin_predictWhole()
			else: 
				print("Running GradientBoostingRegressor...")
				print("Accuracy: {}".format(clff))
				grad_predictWhole()
		else:
			print("Either args.val_time/args.val_prec is not defined OR whole_day is not defined! Define them using --val_time or --val_prec.")

if __name__ == "__main__":
	main()