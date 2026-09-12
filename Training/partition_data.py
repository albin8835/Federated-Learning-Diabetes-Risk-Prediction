import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def partition_dataset(dataset_path="diabetes_prediction_dataset.csv", num_clients=4, alpha=0.5, non_iid=True, random_state=42):
    print("=" * 70)
    print("    FEDERATED LEARNING DATA PARTITIONING & HOSPITAL SIMULATION")
    print("=" * 70)
    print(f"Configuration: Clients={num_clients} | Non-IID={non_iid} (Dirichlet Alpha={alpha})")
    
    # 1. Load and clean raw dataset
    df = pd.read_csv(dataset_path).drop_duplicates()
    
    # Categorical mapping
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    smoking_map = {"never": 0, "No Info": 1, "current": 2, "former": 3, "ever": 4, "not current": 5}
    
    df["gender"] = df["gender"].map(gender_map)
    df["smoking_history"] = df["smoking_history"].map(smoking_map)
    
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"].values
    
    # 2. Train / Test Split (Holdout 20% for Global Evaluation)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 3. Fit Standard Scaler on Training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    # Save Scaler & Global Test Data
    dump(scaler, "scaler.pkl")
    pd.DataFrame(X_test_scaled).to_csv("X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv", index=False)
    print(f"Global Test Set Saved: {len(X_test_scaled)} samples (Positive: {np.sum(y_test == 1)})")
    
    # 4. Partition Training Data across Clients
    np.random.seed(random_state)
    client_indices = [[] for _ in range(num_clients)]
    
    if non_iid:
        # Non-IID Partitioning using Dirichlet Distribution over label classes
        for class_val in [0, 1]:
            class_indices = np.where(y_train == class_val)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            splits = np.split(class_indices, split_points)
            for cid in range(num_clients):
                client_indices[cid].extend(splits[cid])
    else:
        # Standard IID Partitioning (Uniform Random)
        all_indices = np.arange(len(y_train))
        np.random.shuffle(all_indices)
        splits = np.array_split(all_indices, num_clients)
        for cid in range(num_clients):
            client_indices[cid] = splits[cid].tolist()
            
    # 5. Save Client CSVs & Compile Summary Table
    client_stats = []
    
    for cid in range(num_clients):
        c_idx = np.array(client_indices[cid])
        np.random.shuffle(c_idx)
        
        Xc = X_train_scaled[c_idx]
        yc = y_train[c_idx]
        
        # Save client CSVs
        pd.DataFrame(Xc).to_csv(f"client_{cid+1}_X.csv", index=False)
        pd.DataFrame(yc).to_csv(f"client_{cid+1}_y.csv", index=False)
        
        pos_count = int(np.sum(yc == 1))
        total_count = len(yc)
        pos_rate = (pos_count / total_count * 100) if total_count > 0 else 0
        
        client_stats.append({
            "Hospital / Client": f"Hospital Client {cid+1}",
            "Total Samples": total_count,
            "Diabetic Cases": pos_count,
            "Non-Diabetic Cases": total_count - pos_count,
            "Diabetic Rate (%)": f"{pos_rate:.2f}%"
        })
        
    df_stats = pd.DataFrame(client_stats)
    print("\n" + "=" * 75)
    print("                      HOSPITAL CLIENT DATA SUMMARY")
    print("=" * 75)
    print(df_stats.to_string(index=False))
    print("=" * 75)
    
    # 6. Plot Client Distribution Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    c_names = [f"Hospital {i+1}" for i in range(num_clients)]
    non_diab = [s["Non-Diabetic Cases"] for s in client_stats]
    diab = [s["Diabetic Cases"] for s in client_stats]
    
    bar_width = 0.5
    indices = np.arange(num_clients)
    
    ax.bar(indices, non_diab, bar_width, label="Non-Diabetic", color="#2b6cb0")
    ax.bar(indices, diab, bar_width, bottom=non_diab, label="Diabetic (Positive)", color="#e53e3e")
    
    ax.set_ylabel("Patient Sample Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Client Data Distribution ({'Non-IID (Dirichlet α=' + str(alpha) + ')' if non_iid else 'Uniform IID'})", fontsize=13, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels(c_names, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart_file = "client_data_distribution.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()
    print(f"\nDistribution chart saved to '{chart_file}'")
    print("Partitioning completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition Diabetes Dataset for Federated Learning")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter (lower = more non-IID skew)")
    parser.add_argument("--iid", action="store_true", help="Use uniform IID partitioning instead of Non-IID")
    args = parser.parse_args()
    
    partition_dataset(num_clients=args.clients, alpha=args.alpha, non_iid=(not args.iid))
