from pathlib import Path
import os 
import scipy.io

def split():
    my_dictionary = dict()
    data_path = Path('/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/train_frames')
    list_files= sorted(os.listdir(data_path))
    my_dictionary['baseball_pitch']=[]
    my_dictionary['baseball_swing']=[]
    my_dictionary['bench_press']=[]
    my_dictionary['bowling']=[]
    my_dictionary['clean_and_jerk']=[]
    my_dictionary['golf_swing']=[]
    my_dictionary['jumping_jacks']=[]
    my_dictionary['pushups']=[]
    my_dictionary['pullups']=[]
    my_dictionary['situp']=[]
    my_dictionary['squats']=[]
    my_dictionary['tennis_forehand']=[]
    my_dictionary['tennis_serve']=[]
    #my_dictionary['strum_guitar']=[]
    #my_dictionary['jump_rope']=[]
    
    pos_path = os.listdir("/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/penn_action_labels/train")
    my_dictionary.keys()
    breakpoint()

    for i in my_dictionary.keys():
        element=list_files[i]
        labels = scipy.io.loadmat(f'/home/sara/Project/Positional_Diffusion/datasets/Penn_Action/labels/{element}.mat')
        act= ['baseball_pitch', 'clean_and_jerk','pullup','strum_guitar','baseball_swing','golf_swing','pushup','tennis_forehand' ,'bench_press','jumping_jacks','situp','tennis_serve','bowl','jump_rope','squat']
        label = labels['action']
        if label!='strum_guitar'and label!='jump_rope':
            idx=act.index(label)
            my_dictionary[act[idx]].append(element)
        #if label[0]=='baseball_pitch':
           # my_dictionary['baseball_pitch'].append(element)
        #if label[0] == 'baseball_swing':
            #my_dictionary['baseball_pitch'].append(element)
        
      


    return my_dictionary

if __name__ == "__main__":
    dizionario=split()
    breakpoint()