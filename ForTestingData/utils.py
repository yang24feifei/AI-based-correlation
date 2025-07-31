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
from dataloader import DataGenerator
from datetime import datetime
import torch.optim as optim
import tempfile
import gc
from collections import Counter


from sklearn.metrics.pairwise import cosine_distances

def sample_fullcluster( vector_set, context_mask,ration=0.01):
# for the too large vecs to cluster:
    sample_ratio = ration  # e.g., 1%
    sampled_indices = []
    
    # 1. choose sample
    # for _, context_mask in indices_y[:1]:
    indices = context_mask.nonzero(as_tuple=True)[0] if context_mask.dtype == torch.bool else context_mask
    n_sample = max(1, int(len(indices) * sample_ratio))
    sampled_indices = indices[torch.randperm(len(indices))[:n_sample]]
    # sampled_indices.append(sampled)
    
    # sampled_indices = torch.cat(sampled_indices)
    sampled_vectors = vector_set[sampled_indices]  #.cpu().numpy()

    #2. cluster samples
    sampled_labels = interpreter.dbscan.dbscan(
        X=sampled_vectors,
        eps=interpreter.eps,
        min_samples=interpreter.min_samples,
        verbose=True,
    )
    # Compute cluster centroids
    valid = sampled_labels != -1
    sampled_centroids = []
    sampled_cluster_ids = []
    
    for label in np.unique(sampled_labels[valid]):
        mask = (sampled_labels == label)
        centroid = sampled_vectors[mask].mean(axis=0)
        sampled_centroids.append(centroid)
        sampled_cluster_ids.append(label)
    
    sampled_centroids = np.vstack(sampled_centroids)  # shape: (k, dim)

    sampled_centroids = np.asarray(sampled_centroids)


# for _, context_mask in indices_y[:1]:
    indices = context_mask.nonzero(as_tuple=True)[0] if context_mask.dtype == torch.bool else context_mask
    vecs = vector_set[context_mask]  #.cpu().numpy()

    # Compute cosine distances to centroids
    distances = cosine_distances(vecs, sampled_centroids)
    nearest = np.argmin(distances, axis=1)

    # Assign labels based on nearest centroid
    labels = np.array([sampled_cluster_ids[i] for i in nearest])
    # result[indices] = torch.from_numpy(labels)

    return labels


    
def process_large_data_CB_fit(big_contexts, big_events,  big_labels, args,model=None):
    """
    Process a large dataset in chunks and combine the results.
    
    Args:
        features: Large feature tensor/array
        labels: Labels tensor/array
        batch_size: Batch size for DataLoader
        chunk_size: Size of each chunk to process
        model: PyTorch model to use for inference
        device: Device to run the model on ('cuda' or 'cpu')
        args: Namespace object containing parameters
    Returns:
        Combined output from processing all chunks
    """
    # batch_size=args.dataloader_batch_size
    chunk_size=args.dataloader_chunk_size  # *3 #(H100) # // 4
    
    # Get the total size of the dataset
    total_size = len(big_events)
    
    # Calculate the number of chunks
    num_chunks = (total_size + chunk_size - 1) // chunk_size
        
    # Process each chunk
    for i in range(num_chunks):
        # Calculate start and end indices for this chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        
        print(f"Processing chunk {i+1}/{num_chunks} (indices {start_idx} to {end_idx})")
        
        # Get subset of features and labels for this chunk
        chunk_contexts = big_contexts[start_idx:end_idx]
        chunk_events = big_events[start_idx:end_idx]
        chunk_labels = big_labels[start_idx:end_idx]
        
        # # Create a DataGenerator and DataLoader for this chunk
        # chunk_generator = DataGenerator(chunk_features, chunk_labels)
        # chunk_loader = torch.utils.data.DataLoader(chunk_generator, batch_size=batch_size, shuffle=False)

        # Train the ContextBuilder
        model.fit(
            X             = chunk_contexts,               # Context to train with
            y             = chunk_events.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
            labels        = chunk_labels,
            epochs        = args.epochs_CB,                          # Number of epochs to train with
            batch_size    = args.batch_CB,                         # Number of samples in each training batch, in paper this was 128
            learning_rate = args.learning_rate_CB,     # Learning rate to train with, in paper this was 0.01
            verbose       =  args.silent, # not args.silent,                        # If True, prints progress
            optimizer=optim.Adamax, #optim.SGD,
            teach_ratio=0.5
        )

    return model
    
def convert2level( scores, plot=False):
    ''' following DeepCASE, set 5 risk categories.
    '''
    if plot:
        # Step 1: Plot histogram
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
        plt.title("Histogram of Reconstruction Scores")
        plt.xlabel("Reconstruction Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        f=plt.gcf()
        f.savefig('result/risklevel.pdf')
        f.clear()
        plt.show()
    
    # Step 2: Statistical Thresholding
    mu = np.mean(scores)
    sigma = np.std(scores)
    
    print(f"Mean (μ): {mu:.4f}")
    print(f"Std Dev (σ): {sigma:.4f}")
    
    # Step 3: Define risk assignment
    def assign_risk(score, mu, sigma):
        if score == mu:
            return 0 # "INFO"
        elif mu < score <= (mu + sigma) or (mu-sigma)<= score< mu:
            return 1 # "LOW"
        elif (mu + sigma) < score <= (mu + 2 * sigma) or (mu-sigma*2) <= score < (mu-sigma):
            return 2 #"MEDIUM"
        elif mu + (2 * sigma) < score <= (mu + 3 * sigma) or (mu-3*sigma)<= score< ( mu-(2*sigma)):
            return 3 #"HIGH"
        elif score > mu + 3 * sigma or score< mu -3*sigma:
            return 4 #"ATTACK"
        else:
            return 0 # "INFO"  # covers edge cases
    
    # Step 4: Assign risk levels
    risk_levels = [assign_risk(s, mu, sigma) for s in scores]
    
    # Step 5: Print unique risk levels and counts
    risk_summary = Counter(risk_levels)
    print("\nRisk Level Summary:")
    for level, count in risk_summary.items():
        print(f"{level}: {count}")

    return risk_levels  

    
def find_best_threshold(folder_path, target_datafile):
    # find threshold from history the best result file 
    best_auc = -1
    best_threshold = None

    for filename in os.listdir(folder_path):
        if filename.startswith('metrics') and filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath)
                # Filter rows where datafile matches the target
                filtered = df[df['datafile'] == target_datafile]
                
                if not filtered.empty:
                    # Find the row with max AUCROC
                    max_row = filtered.loc[filtered['AUCROC'].idxmax()]
                    if max_row['AUCROC'] > best_auc:
                        best_auc = max_row['AUCROC']
                        best_threshold = max_row['threshold']
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return best_threshold

####################################################################################    
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
    
def process_large_data_Interp_predict(big_contexts, big_events, args, model=None):
    chunk_size = args.dataloader_chunk_size    // 2
    total_size = len(big_events)
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    temp_dir = tempfile.mkdtemp()  # Creates a temporary directory
    result1_paths, result2_paths = [], []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        print(f"Processing chunk {i+1}/{num_chunks} (indices {start_idx} to {end_idx})")

        chunk_contexts = big_contexts[start_idx:end_idx]
        chunk_events = big_events[start_idx:end_idx]

        # with torch.no_grad():
        chunk_pred, chunk_index = model.predict(
            X=chunk_contexts, 
            y=chunk_events.reshape(-1, 1),
            iterations = 10,  # 100,                        # Number of iterations to use for attention query, in paper this was 100
            batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
            verbose    = True,                       # If True, prints progress
        )

        # Save chunk results as .pt files
        path1 = os.path.join(temp_dir, f"chunk_pred_{i}.pt")
        path2 = os.path.join(temp_dir, f"chunk_index_{i}.pt")
        torch.save(chunk_pred, path1)
        torch.save(chunk_index, path2)
        result1_paths.append(path1)
        result2_paths.append(path2)

        # Clear memory
        del chunk_contexts, chunk_events, chunk_pred, chunk_index
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load and concatenate all results
    all_result1 = [torch.load(p, weights_only=False) for p in result1_paths]
    all_result2 = [torch.load(p, weights_only=False) for p in result2_paths]

    final_result1 = np.concatenate(all_result1, axis=0)
    final_result2 = np.concatenate(all_result2, axis=0)

    # Clean up temporary files
    for p in result1_paths + result2_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    return final_result1, final_result2

  

def process_large_data_CB_predict(big_contexts, big_events, args, model=None):
    chunk_size = args.dataloader_chunk_size //3 #A100   // 2 #(h100)
    total_size = len(big_events)
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    temp_dir = tempfile.mkdtemp()  # Creates a temporary directory
    result1_paths, result2_paths = [], []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        print(f"Processing chunk {i+1}/{num_chunks} (indices {start_idx} to {end_idx})")

        chunk_contexts = big_contexts[start_idx:end_idx]
        chunk_events = big_events[start_idx:end_idx]

        with torch.no_grad():
            chunk_confi, chunk_atten = model.predict(chunk_contexts, chunk_events.reshape(-1, 1))

        # Save chunk results as .pt files
        path1 = os.path.join(temp_dir, f"chunk_confi_{i}.pt")
        path2 = os.path.join(temp_dir, f"chunk_atten_{i}.pt")
        torch.save(chunk_confi.cpu(), path1)
        torch.save(chunk_atten.cpu(), path2)
        result1_paths.append(path1)
        result2_paths.append(path2)

        # Clear memory
        del chunk_contexts, chunk_events, chunk_confi, chunk_atten
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load and concatenate all results
    all_result1 = [torch.load(p) for p in result1_paths]
    all_result2 = [torch.load(p) for p in result2_paths]

    final_result1 = torch.cat(all_result1, dim=0)
    final_result2 = torch.cat(all_result2, dim=0)

    # Clean up temporary files
    for p in result1_paths + result2_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    return final_result1, final_result2

def threshold_search_confidence(y_true, y_confi, y_pred, plot=False):
    best_threshold = 0
    best_score = 0

    threshold_list=[]
    accuracy_list=[]
    f1_ist=[]

    for rate in np.arange(0.01,1, 0.01):
        threshold=np.quantile(y_confi,rate)
        y_mask=y_confi > threshold
        metric_report=classification_report(y_true[y_mask], y_pred[y_mask],output_dict=True)       
        try :
            f1_positive=metric_report['weighted avg']['f1-score']
        except:
            f1_positive=0

        threshold_list.append(threshold)
        accuracy_list.append(metric_report['accuracy'])
        f1_ist.append(metric_report['weighted avg']['f1-score'])

        if metric_report['accuracy'] >best_score and f1_positive!=0 : 
            best_threshold = threshold
            best_score = metric_report['accuracy']   # metric_report['weighted avg']['f1-score']

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(threshold_list, accuracy_list, label='Accuracy score')
        plt.plot(threshold_list, f1_ist, label='F1 Score')
        plt.xlabel('Condidence Threshold')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        f=plt.gcf()
        f.savefig('result/confidencesearch.pdf')
        f.clear()
        plt.show()

    return best_score, best_threshold

def load_anomaly_results(csv_file_path):
    """
    Load anomaly scores and labels from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file containing anomaly results
        
    Returns:
        Tuple of (anomaly_scores, anomaly_labels, anomaly_levels) as numpy arrays
    """
    try:
        print(f"Loading anomaly results from {csv_file_path}")
        
        # Read the CSV file into a pandas DataFrame
        results_df = pd.read_csv(csv_file_path)
        
        # Extract the columns
        if 'anomaly_score' in results_df.columns and 'anomaly_label' in results_df.columns:
            anomaly_scores = results_df['anomaly_score'].values
            anomaly_labels = results_df['anomaly_label'].values
            anomaly_levels = results_df['anomaly_level'].values
            
            print(f"Successfully loaded {len(anomaly_scores)} records")
            print(f"Anomaly distribution: {np.sum(anomaly_labels)} anomalies, {len(anomaly_labels) - np.sum(anomaly_labels)} normal")
            
            print(f"Successfully loaded {len(anomaly_levels)} records")
            
            return anomaly_scores, anomaly_labels, anomaly_levels
        else:
            missing_cols = []
            if 'anomaly_score' not in results_df.columns:
                missing_cols.append('anomaly_score')
            if 'anomaly_label' not in results_df.columns:
                missing_cols.append('anomaly_label')
            if 'anomaly_level' not in results_df.columns:
                missing_cols.append('anomaly_level')
                
            print(f"Error: CSV file missing required columns: {', '.join(missing_cols)}")
            print(f"Available columns: {', '.join(results_df.columns)}")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None, None, None        

def save_results_to_csv(anomaly_scores, anomaly_labels, anomaly_levels, output_file='result/anomaly_results.csv'):
    """
    Save anomaly scores and labels to a CSV file.
    
    Args:
        anomaly_scores: Numpy array of anomaly scores
        anomaly_labels: Numpy array of anomaly labels (0 for normal, 1 for anomaly)
        output_file: Path to save the CSV file
    """
    print(f"Saving results to {output_file}")
    
    # If anomaly_scores has multiple columns, we might need to process them
    if len(anomaly_scores.shape) > 1 and anomaly_scores.shape[1] > 1:
        # If model outputs multiple values, take the max or first value as score
        anomaly_scores = anomaly_scores[:, 0]
    
    # Create a DataFrame
    results_df = pd.DataFrame({
        'anomaly_score': anomaly_scores.flatten(),
        'anomaly_label': anomaly_labels.flatten(),
        'anomaly_level': anomaly_levels
    })
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved successfully to {output_file}")
    
def process_large_data_SD(features, labels,  args, model=None):
    """
    Process a large dataset in chunks and combine the results.
    
    Args:
        features: Large feature tensor/array
        labels: Labels tensor/array
        batch_size: Batch size for DataLoader
        chunk_size: Size of each chunk to process
        model: PyTorch model to use for inference
        device: Device to run the model on ('cuda' or 'cpu')
        args: Namespace object containing parameters
    Returns:
        Combined output from processing all chunks
    """
    batch_size=args.dataloader_batch_size
    chunk_size=args.dataloader_chunk_size
    device=args.device
    
    # Get the total size of the dataset
    total_size = len(features)
    
    # Calculate the number of chunks
    num_chunks = (total_size + chunk_size - 1) // chunk_size
    
    # Initialize list to store outputs
    all_scores = []
    
    # Process each chunk
    for i in range(num_chunks):
        # Calculate start and end indices for this chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        
        print(f"Processing chunk {i+1}/{num_chunks} (indices {start_idx} to {end_idx})")
        
        # Get subset of features and labels for this chunk
        chunk_features = features[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        # Create a DataGenerator and DataLoader for this chunk
        chunk_generator = DataGenerator(chunk_features, chunk_labels)
        chunk_loader = torch.utils.data.DataLoader(chunk_generator, batch_size=batch_size, shuffle=False)
        
        # Process this chunk (collect outputs)        
        with torch.no_grad():  
            y_true=[]
            en_list, ex_list, re_en_list, re_ex_list=[],[],[],[]
            for batch_idx, (xx,y) in enumerate(tqdm(chunk_loader)): 
                content_true=xx[:,:args.embed_content_dim]
                context_true=xx[:,-args.length:].long()
                x=xx[:,:-args.length]
                x = x.to(device, dtype=torch.float32)
                # y = y.to(device, dtype=torch.float32)
                
                recon_content, recon_context = model(x)  
                
                y_true.append(y)
                en_list.append(content_true)
                ex_list.append(context_true)
                re_en_list.append(recon_content)
                re_ex_list.append(recon_context)
            
        re_en_list = torch.cat(re_en_list).cpu()
        re_ex_list = torch.cat(re_ex_list).cpu()
        
        en_list=torch.cat(en_list)
        ex_list=torch.cat(ex_list)
        y_true=torch.cat(y_true)
        content_loss = F.mse_loss(re_en_list, en_list)
        context_loss = F.cross_entropy(re_ex_list.reshape(-1, re_ex_list.size(-1)), ex_list.reshape(-1), reduction='none')
        loss= args.alpha_loss* context_loss+ (1-args.alpha_loss) *content_loss    
        loss = loss.reshape(ex_list.size(0), ex_list.size(1))  # [seq_len, window_size]
        seq_score=loss.mean(dim=1).reshape(-1,1).detach().numpy()
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()    
        chunk_score = scaler.fit_transform(seq_score)
        all_scores.append(chunk_score)
        
        # Clear memory
        del chunk_features, chunk_labels, chunk_generator, chunk_loader
        del y_true, en_list, ex_list, re_en_list, re_ex_list, 
        # content_ture, context_true, recon_content, recon_context
        del loss, content_loss, context_loss, seq_score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all chunks
    if all_scores:
        final_scores = np.concatenate(all_scores)
        return final_scores
    else:
        return None
        

#################################################################################
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
        f=plt.gcf()
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