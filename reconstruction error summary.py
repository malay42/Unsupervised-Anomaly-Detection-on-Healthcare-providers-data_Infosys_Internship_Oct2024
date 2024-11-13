print("\nReconstruction Error Summary:")
for method_name, error_column in [
    ('Isolation Forest', 'ISO_Reconstruction_Error'),
    ('DBSCAN', 'DBSCAN_Reconstruction_Error'),
    ('KNN (LOF)', 'KNN_Reconstruction_Error'),
    ('Gaussian Mixture Model', 'GMM_Reconstruction_Error'),
    ('K-Means', 'KMeans_Reconstruction_Error')
]:
    print(f"{method_name}: Mean Reconstruction Error = {data_dropped[error_column].mean()}")