import json
import os
import pandas as pd
import ipaddress
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from sklearn.metrics import classification_report
from sklearn import metrics

from datetime import datetime

# Create or append metrics to CSV file
def log_metrics_to_csv(metrics: dict, csv_path: str = "result/metrics.csv"):
    # Add optional run metadata
    metrics["timestamp"] = datetime.now().isoformat()

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics])

    # Append to CSV
    if os.path.exists(csv_path):
        metrics_df.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(csv_path, index=False, header=True)
        
#plot the metrics(FP,TP,ROC,AUC)
def rocauc(attackname,true_y, pre_y,plot=True):
    '''  y_true:真实值
    y_score：预测概率。注意：不要传入预测label！！！
    '''
    fpr,tpr,threshold=metrics.roc_curve(true_y,pre_y)
    
    roc_auc=metrics.auc(fpr,tpr) 
    try:
        roc_auc=float(roc_auc)
    except:
        print(f'float convert error:{roc_auc}')
        roc_auc=0
    if np.isnan(roc_auc):
        roc_auc=0
        
    opti_point=np.argmax(tpr-fpr)
    if plot:
        plt.figure(figsize=(6,6))
        plt.title('Validation ROC-%s'%attackname)
        plt.plot(fpr,tpr,'b',label='Val AUC=%0.3f'%roc_auc)
        plt.plot(fpr[opti_point],tpr[opti_point],marker='o',color='r',label='bes-thre=%0.3f'%threshold[opti_point])
        plt.legend(loc='lower right')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim([0,1])
        plt.ylim([0,1])
        '''
        f=pyplot.gcf()
        f.savefig(attackname+'-rocauc.pdf')
        f.clear()
        ''' 
        plt.show()
        
    return threshold[opti_point], fpr[opti_point] ,roc_auc,opti_point 

def  auc_pr_func(y_true, y_scores,plot=True):
    '''
    auc under precision and recall. suit to extremly imbalanced dataset.
    '''
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    # Find best threshold (max F1)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    
    # Convert scores to binary predictions using the best threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    
    # Compute confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute FPR = FP / (FP + TN)
    fpr = fp / (fp + tn + 1e-8)  # small epsilon to avoid divide-by-zero
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"False Positive Rate (FPR) at best threshold: {fpr:.4f}")
    
    if plot:
        # Plot PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC-PR = {auc_pr:.4f})')
        plt.scatter(recall[best_idx], precision[best_idx], color='red', zorder=5,
                    label=f'Best Threshold = {best_threshold:.2f}\nF1 = {best_f1:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Best Threshold')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return best_threshold, fpr ,auc_pr, best_f1


####################### convert json to csv & align timestamp  #######################################################
def json2csv_timestamp(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)
            csv_path = os.path.join(output_folder, filename.replace(".json", ".csv"))
    
            try:
                data=[]
                #  Load JSON file
                with open(json_path, "r", encoding="utf-8", buffering=1) as f:
                    for line in f:
                        line_json=json.loads(line.strip())
    
                        # unify the format of timestamp that all with milliseconds
                        if '.' not in line_json["timestamp"]:  # Check if milliseconds are missing
                            temp = line_json["timestamp"].rsplit('-', 1)  # Split from the last '-' (timezone part)
                            line_json["timestamp"] = f"{temp[0]}.000-{temp[1]}"  # Add '.000' before timezone
    
                        data.append(line_json)  # Load each line as JSON
                    f.flush()
    
                #  Convert JSON to DataFrame
                df = pd.json_normalize(data, sep="_")  # Flatten nested JSON
    
                #  Save DataFrame to CSV
                df.to_csv(csv_path, index=False)
                print(f"Converted: {filename} → {csv_path}")
    
            except Exception as e:
                print(f"Error converting {filename}: {e}")


########################## append ID and run function  ############################################################

def is_multicast(ip):
    """
    Check if an IP address is in the multicast range.
    Works for both IPv4 and IPv6.
    return:
    'mcast': multicast
    'scast': singlecast
    """
    try:
        if ipaddress.ip_address(ip).is_multicast:
            return "mcast"
        else: 
            return "scast"
    except ValueError:
        return None  # Handles invalid/missing IPs


''' In PROCESS events, the image_path field shows where the process executable is located. 
 classify processes as 'system' or 'user' based on their file path; and 'critical' when visit security file; 'unknown' for other scenes.
'''

# Define system path prefixes for Windows [, Linux, and macOS]
SYSTEM_PATHS = [
    # Windows
    "Windows\\System32", "Windows\\SysWOW64", "Program Files\\WindowsApps",
    # # Linux
    # "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/lib/systemd",
    # # macOS
    # "/System/Library", "/usr/bin", "/usr/sbin", "/Library"
]

#  Special Cases**
# Security-Relevant Events**:
# Define critical security files (Windows [, Linux, macOS] )
CRITICAL_FILES = [
    # Windows Event Logs
    "Windows\\System32\\winevt\\Logs\\Security.evtx",
    "Windows\\System32\\config\\SAM",
    "Windows\\System32\\config\\SECURITY",
    # # Linux Log Files
    # "/var/log/auth.log", "/var/log/secure", "/etc/shadow",
    # # macOS Log Files
    # "/var/log/system.log", "/private/var/log/asl/"
]

def classify_process(image_path):
    """Classify a process as 'system' or 'user' based on image_path."""
    if pd.isna(image_path):  # Handle missing values
        return "unknown"
    
    # Normalize case for Windows paths
    image_path = str(image_path).strip().lower()
    
    # Check if the process path starts with a known system directory
    if any(path.lower() in image_path for path in SYSTEM_PATHS):
        if any(file.lower() in image_path for file in CRITICAL_FILES):
             return "critical"
        return "system"
    
    return "user"


#Unusual process chains (e.g., `PING.EXE` spawning `Explorer.EXE` → `SuspiciousProcessSpawn`).
# Define suspicious parent-child process pairs
SUSPICIOUS_PROCESS_CHAINS = [
    (r"ping.exe", r"explorer.exe"),
    (r"cmd.exe", r"powershell.exe"),
    (r"winword.exe", r"cmd.exe"),
    (r"rundll32.exe", r"cmd.exe"),
    (r"lsass.exe", None)  # LSASS should never spawn a child process
]

# Detect suspicious process chains
def is_suspicious_spawn(parent, child):
    """Check if a parent-child process pair is suspicious, handling missing values."""

    parent = parent.lower().strip() if pd.notna(parent) else ""  # Normalize slashes
    child = child.lower().strip() if pd.notna(child) else ""

    for suspicious_parent, suspicious_child in SUSPICIOUS_PROCESS_CHAINS:
        if  suspicious_parent in parent:
            if suspicious_child is None or  suspicious_child in child:
                return "suspicious"
    return "nonsusp"


# extract domain/group name
def extract_username(rawname):
    """extract high one level name for user/user_name/task_name."""
    if pd.isna(rawname):
        return None
    elif '\\' in rawname:
        return rawname.split('\\')[-2].replace(' ','').strip().lower()
    elif '-' in rawname:
        return rawname.split('-')[-1].replace(' ','').strip().lower()
    return rawname.replace(' ','').strip().lower()

def extract_taskname(rawname):
    if pd.isna(rawname):
        return None
    elif '\\' in rawname:        
        if '{' and '}' and '-' in rawname:
            return rawname.split('-')[-2].replace(' ','').strip().lower()        
        return rawname.split('\\')[-1].replace(' ','').strip().lower()
    elif '-' in rawname:
        return rawname.split('-')[-1].replace(' ','').strip().lower()
        
    return rawname.replace(' ','').strip().lower()



##########################################################################################################
    
def addIndex4Uniq(readname):
    shortlist=pd.read_csv(readname)

    # Create dictionary mapping index to unique events
    event_dict = {i: event for i, event in enumerate(np.unique(shortlist["event"].values))}
    
    # Convert dictionary to DataFrame
    df_event = pd.DataFrame(list(event_dict.items()), columns=["id", "event"])
    
    # Save to CSV
    savename=readname.split('.')[0] +'Index'+'.csv'
    if os.path.exists(savename):
        file_name=savename.split('.')[0] +"-1"+savename.split('.')[1] 
        df_event.to_csv(file_name, index=False)
    else:
        df_event.to_csv(savename, index=False)
    
    print(f"Saved unique events and index to {savename}")


    
def reConstructData( file_folder, extract_folder, short_file):
    '''
extract out the specific fields name and the values. prepare data for BERT.
1. base: the value of action+object, fields name( the fields name while the value is not empty), fields value( joint all the non empyt value of fields)
2. enhance: the value of action+object +pid+principal+tid, fields name and value as above

['action', 'actorID', 'hostname', 'id', 'object', 'objectID', 'pid', 'ppid', 'principal', 'tid', 'timestamp', 
'properties_acuity_level', 'properties_image_path', 'properties_src_pid', 'properties_src_tid', 
'properties_stack_base', 'properties_stack_limit', 'properties_start_address', 'properties_subprocess_tag', 
'properties_tgt_pid', 'properties_tgt_tid', 'properties_user_stack_base', 'properties_user_stack_limit', 
'properties_dest_ip', 'properties_dest_port', 'properties_direction', 'properties_l4protocol',
'properties_src_ip', 'properties_src_port', 'properties_size', 'properties_file_path', 'properties_info_class',
'properties_new_path', 'properties_end_time', 'properties_start_time', 'properties_parent_image_path',
'properties_command_line', 'properties_data', 'properties_key', 'properties_type', 'properties_value', 
'properties_base_address', 'properties_module_path', 'properties_tgt_pid_uuid', 'properties_sid', 
'properties_user']
'''
    # Process all CSV files in the folder
    for file_name in os.listdir(file_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(file_folder, file_name)        
            shortlist=[]   
            
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                ################################## prepare short/ event type ################################################
                short=''
                # add base event short string
                short=f"{row['object'] if pd.notna( row['object']) else 'none'}_{row['action'] if pd.notna( row['action']) else 'none'}"
        
                # add enhanced event short string    
                if row["object"]=="FLOW":
                    multicast= is_multicast( row["properties_dest_ip"] ) 
                    short+=f"_{row['properties_direction'].replace(' ','').strip().lower() if pd.notna( row['properties_direction']) else 'none'}_{str(row['properties_l4protocol']).split('.')[0].replace(' ','').strip().lower() if pd.notna( row['properties_l4protocol']) else 'none'}_{multicast}"
                elif row["object"]=="FILE":
                    if row["action"]=="CREATE" or row["action"]=="READ" or row["action"]=="RENAME" or row["action"]=="WRITE":                
                        short+=f"_{classify_process(row['properties_image_path']) }"
                    elif row["action"]=="DELETE" or row["action"]=="MODIFY":
                        short+=f"_{classify_process(row['properties_image_path'])}_{row['properties_info_class'].replace(' ','').strip().lower() if pd.notna( row['properties_info_class']) else 'none'}"
                elif row["object"]=="PROCESS":
                    if row["action"]=="CREATE" or row["action"]=="TERMINATE":
                        short+=f"_{extract_username( row['properties_user'])}"     
                    elif row["action"]=="OPEN":
                        short+=f"_{is_suspicious_spawn(row['properties_parent_image_path'], row['properties_image_path'])}"
                elif row["object"]=="REGISTRY":
                    if row["action"]=="ADD" or row["action"]=="EDIT":
                        short+=f"_{row['properties_type'].split('_')[-1].replace(' ','').strip().lower() if pd.notna( row['properties_type']) else 'none'}"
                # elif row["object"]=="SERVICE":
                #     short+=f"_{row["service_type"].replace(' ','').strip().lower() if pd.notna( row["service_type"]) else "none"}"
                elif row["object"]=="TASK":
                    if row["action"]=="CREATE" or row["action"]=="DELETE" or row["action"]=="MODIFY":
                        short+=f"_{extract_taskname( row['properties_task_name']) }_{extract_username(row['properties_user_name'])}"
                elif row["object"]=="USER_SESSION":
                    # if row["action"]=="INTERACTIVE" or row["action"]=="LOGIN" or row["action"]=="LOGOUT" or row["action"]=="REMOTE":
                    short+=f"_{row['properties_requesting_domain'].replace(' ','').strip().lower() if pd.notna( row['properties_requesting_domain']) else 'none'}"
                 
                shortlist.append(short)  
                
            df["event"]=shortlist
                
            # update distinct event type 
            short_distinct_new=pd.DataFrame(set(shortlist), columns=["event"])  
            if os.path.exists(short_file):
                short_distinct_old=pd.read_csv(short_file)
                short_distinct=pd.concat([short_distinct_old, short_distinct_new], ignore_index=True).drop_duplicates(ignore_index=True)
                short_distinct.to_csv(short_file, index=False)
            else:
                short_distinct_new.to_csv(short_file, index=False)
    
           
            ###################################### prepare fields and values #############################
            # Identify columns that start with 'properties_'
            properties_cols = [col for col in df.columns if col.startswith("properties_")]        
            # Create merged values column (excluding empty or None values)
            df["field_values"] = df[properties_cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)
            # Create merged column names (excluding 'properties_' prefix)
            df["field_names"] = df[properties_cols].apply(lambda row: ', '.join([col.replace("properties_", "") for col, val in row.items() if pd.notna(val) and val != '']), axis=1)
            
            # Drop original properties_ columns
            df = df.drop(columns=properties_cols)         
            
            #######################################################################################      
            if os.path.exists(os.path.join(extract_folder, file_name)):
                pos=file_name.find('.csv')
                file_name=file_name[:pos] +"-1" + file_name[pos:]
            df.to_csv(os.path.join(extract_folder, file_name), index=False)
    
            print("finish on "+os.path.join(extract_folder, file_name))
            
    
    print(f"All CSV files updated successfully. Stored in {extract_folder} and event type in {short_file}")



# get eventID, event mapping from existed csv file
def get_event_mapping(shortfile):
    df=pd.read_csv(shortfile, usecols=["id", "event"])
    # Convert to dictionary
    event_dict = dict(zip(df["id"], df["event"]))
    return event_dict


def field_name_value_prepare(origdata_fieldname, origdata_fieldvalue, embmodel, event_template):
    #################### 1. embedding the field name ##############################################
    # embeddings shape=[sample, 768]   , the same as 'vector'
    embeddings = embmodel.encode(event_template['EventTemplate'].tolist()) 
    event_template['nameVector'] = list(embeddings)
    template_dict = event_template.set_index('EventTemplate')['nameVector'].to_dict()
    
    # convert templates to vectors for all logs
    fieldname_em = []
    for idx, template in enumerate(origdata_fieldname):
        try:
            fieldname_em.append(template_dict[template])
        except KeyError:
            # new template
            fieldname_em.append(embmodel.encode(template))
            
    fieldname_em=torch.tensor( np.vstack( fieldname_em) , dtype=torch.float32)
    
    #################### 2. embedding the field value #############################################
    unique_values = np.unique(origdata_fieldvalue)  # Get unique values
    
    value_to_index = {v: idx for idx, v in enumerate(unique_values)}  # Map each unique value to an index
    fieldva_ind = torch.tensor([value_to_index[v] for v in origdata_fieldvalue])
    fieldva_ind=fieldva_ind.unsqueeze(1)

    return fieldname_em,fieldva_ind


# def tune_threshold(y_true, y_pred_raw, average='weighted', steps=101,plot=True):
#     """
#     Tune the threshold for binary classification and plot metrics vs threshold.

#     Args:
#         y_true (np.ndarray): Ground truth labels (0/1).
#         y_pred_raw (np.ndarray): Raw model outputs (probabilities or logits).
#         average (str): Averaging method for scoring. E.g., 'weighted', 'macro', 'binary'.
#         steps (int): Number of threshold steps between 0 and 1.

#     Returns:
#         dict: Contains best threshold, precision, recall, f1, and all metrics per threshold.
#     """

#     thresholds = np.linspace(0, 1, steps)
#     precisions, recalls, f1s = [], [], []

#     best_f1 = -1
#     best_threshold = 0

#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.detach().cpu().numpy()
#     if isinstance(y_pred_raw, torch.Tensor):
#         y_pred_raw = y_pred_raw.detach().cpu().numpy()
    
#     for t in thresholds:
#         pred_bin = (y_pred_raw > t).astype(np.float32)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             y_true, pred_bin, average=average, zero_division=0
#         )

#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = t

#     if plot:
#         # Plotting
#         plt.figure(figsize=(8, 5))
#         plt.plot(thresholds, precisions, label='Precision')
#         plt.plot(thresholds, recalls, label='Recall')
#         plt.plot(thresholds, f1s, label='F1 Score')
#         plt.axvline(best_threshold, linestyle='--', color='gray', label=f'Best Threshold: {best_threshold:.2f}')
#         plt.xlabel('Threshold')
#         plt.ylabel('Score')
#         plt.title(f'Threshold Tuning ({average} average)')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     return {
#         'best_threshold': best_threshold,
#         'precision': precisions[np.argmax(f1s)],
#         'recall': recalls[np.argmax(f1s)],
#         'f1': best_f1,
#         # 'thresholds': thresholds,
#         # 'precisions': precisions,
#         # 'recalls': recalls,
#         # 'f1s': f1s
#     }


# from & for Interpreter
def confusion_report(
        y_true,
        y_pred,
        labels        = None,
        target_names  = None,
        sample_weight = None,
        normalize     = None,
        skip_x        = set(),
        skip_y        = set(),
    ):
    """Print the confusion matrix as a report.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            Actual labels of evaluated values.

        y_pred : array-like of shape=(n_samples,)
            Predicted labels of evaluated values.

        labels : array-like of shape=(n_labels,), optional
            All different labels to include in the confusion report.
            If none are given, labels are inferred from the y_true and y_pred
            inputs. Labels should at least contain all labels in y_true and
            y_pred, but additional labels may be given.

        target_names : array-like of shape=(n_labels,), optional
            If given, use names pressented by target_names for each
            corresponding label.

        sample_weight : array-like of shape=(n_samples,), optional
            If given, weigh input by given sample_weight.

        normalize : {‘true’, ‘pred’, ‘all’}, default=None
            Normalizes confusion matrix over the true (rows), predicted
            (columns) conditions or all the population. If None, confusion
            matrix will not be normalized.

        skip_x : set(), optional
            Set of target_names to skip while printing the columns.

        skip_y : set(), optional
            Set of target_names to skip while printing the rows.

        Returns
        -------
        result : string
            Report detailing the confusion matrix for a given prediction.
        """
    # Compute matrix
    matrix = confusion_matrix(
        y_true        = y_true,
        y_pred        = y_pred,
        labels        = labels,
        sample_weight = sample_weight,
        normalize     = normalize,
    )

    if target_names is not None:
        assert labels       is not None
        assert target_names is not None
        assert len(labels) == len(target_names)

        # Add labels to matrix
        matrix = np.concatenate(([target_names], matrix))
        matrix = np.concatenate(([["T\\P"] + target_names], matrix.T)).T

    # Compute width of rows
    width = np.vectorize(len)(matrix).max()

    # Transform to string
    result = ""
    mask_x = [i for i, x in enumerate(matrix[0   ]) if x not in skip_x]
    mask_y = [i for i, x in enumerate(matrix[:, 0]) if x not in skip_y]
    for row in matrix[mask_y]:
        result += "\t".join(
            "{:>{width}}".format(element, width=width)
            for element in row[mask_x]
        ) + '\n'

    return result

def threshold_search(y_true, y_proba):
    # search threshold for Accuracy
    best_threshold = 0
    best_score = 0
    for rate in np.arange(0.01,1, 0.01):
        threshold=np.quantile(y_proba,rate)
        y_pred=y_proba > threshold
        #score=metrics.f1_score(y_true, y_pred, average='weighted') 
        metric_report=classification_report(y_true, y_pred,output_dict=True)       
        try :
            f1_positive=metric_report['1.0']['f1-score']
        except:
            f1_positive=0
        if metric_report['accuracy'] >best_score and f1_positive!=0 : 
            best_threshold = threshold
            best_score = metric_report['accuracy']   #metric_report['macro avg']['f1-score']
    return best_score, best_threshold