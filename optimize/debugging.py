from composition_prediction import *
import pandas as pd
import matplotlib
import pickle

try:
    with open("./debugging/sent_compositions.pkl", "rb") as file:
        df = pickle.load(file)
except:
    print('Could not read .pkl file')
print(df.shape) # 24 compositions x 3 columns x N number of prints

# df = pd.read_csv("./debugging/sent_compositions_a.csv", index_col=None)
# df = pd.read_csv("sent_compositions_a.csv", index_col=None)





output = []
starts = []
ends=[]
shift=0
zeros = np.zeros(10)
ones = np.ones(10)*23
x=np.arange(0,24)
for n in range(df.shape[2]):
    fig,ax=plt.subplots(1)
    comps = df[:,:,n]
    print(comps.shape)
    # print(comps)
    # print(f"bounds: ({low_bound},{up_bound})")
    start = np.array([comps[0,0],0,0,0,0,0,0,0,comps[0,1], comps[0,2]])
    print(start)
    end = np.array([comps[-1,0],0,0,0,0,0,0,0,comps[-1,1], comps[-1,2]])
    print(end)
    plate='Front'
    max_flow=490
    start_step=0
    grad_interval_time=500/1000
    exp = [start, end, plate, None, max_flow, start_step, grad_interval_time]

    out = deterministic_physics_extraction(exp)
    for i in range(10):
        # print('shapes',x.shape,out[:,i].shape)
        # print('X',x,x.dtype)
        # print('Y',out[:,i],out.dtype)
        ax.scatter(x,out[:,i],s=20,zorder=5,label=f'mod_{i}')
    ax.scatter(zeros,start,color='red')
    ax.scatter(ones,end,color='blue')
    ax.legend()
    fig.savefig(f'./debugging/predicted_composition{n}.png')
    plt.figure(fig.number)
    # plt.show()
    fig.clear()
    output.append(out)
    starts.append(start)
    ends.append(end)
    shift+=1
    # print("collections on ax:", len(ax.collections))
    # print("xlim:", ax.get_xlim(), "ylim:", ax.get_ylim())
    # print(ax.get_facecolor())



output = np.array(output)
output = output.transpose(1,2,0)
starts = np.array(starts)
ends = np.array(ends)
print("Outpuput shape" ,output.shape)
with open('./debugging/actual_compositions.pkl', 'wb') as file:
    pickle.dump(output, file)

print(df.shape, output.shape)

start_df = pd.DataFrame(starts)
start_df.to_csv('./debugging/start_test_compositions.csv')

end_df = pd.DataFrame(ends)
end_df.to_csv('./debugging/end_test_compositions.csv')

# Now Plot and see what was predicted 


# min_error,predicted_comp,max_error = predict_composition(file_name)


# # print(predicted_comp)
# plump=np.zeros((predicted_comp.shape[0],1))

# sql_pred = np.hstack((plump, predicted_comp))
# sql_pred.tolist()
# self.save_to_sql(sql_pred)


