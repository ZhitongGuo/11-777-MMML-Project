import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.models as models
# from torchcam.cams import GradCAM
from sklearn.cluster import KMeans
import cv2
from collections import Counter, defaultdict
# from transformers import ViTFeatureExtractor, ViTImageProcessor
from sklearn.decomposition import PCA


class NormalizeTo01:
    def __call__(self, img):
        # Convert the input image to a Tensor
        img = transforms.ToTensor()(img)

        # Calculate the minimum and maximum values
        min_val = img.min()
        max_val = img.max()

        # Normalize the image to the range [0, 1] using the formula
        img = (img - min_val) / (max_val - min_val)

        return img


def gram_vis(model_name='vgg16', dataset_directory='../search_engine/resources_100k/imgs/'):
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Remove the classifier (fully connected layers) from VGG-16
        model = torch.nn.Sequential(*list(model.children())[:-1])
        feature_size = 4096
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove the classifier (fully connected layer) from ResNet-50
        model = torch.nn.Sequential(*list(model.children())[:-2])
        feature_size = 2048
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Move the model to the GPU
    model.to(device)

    # Set the model to evaluation mode (no gradient computation)
    model.eval()

    # Load the dataset using PyTorch's ImageFolder
    dataset = ImageFolder(dataset_directory, transform=transform)

    # Create a DataLoader to efficiently load and preprocess the images in batches
    batch_size = 1  # Adjust as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize Grad-CAM
    cam = GradCAM(model=model)  # You may need to specify the appropriate target_layer

    all_heatmaps = []
    with torch.no_grad():
        for images, _ in dataloader:
            # Move images to GPU
            images = images.to(device)

            # Forward pass to execute the model
            _ = model(images)  # Ensure forward pass is executed

            # Generate Grad-CAM heatmap
            heatmap = cam(images)

            # Convert the heatmap to a NumPy array
            heatmap_np = heatmap.squeeze().cpu().numpy()

            # Visualize the heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap_np, cmap='viridis')
            plt.axis('off')
            plt.title('Grad-CAM Heatmap')
            plt.show()

            break
            # Store the heatmap (you can process or visualize it as needed)
            all_heatmaps.append(heatmap_np)

    # Extract Grad-CAM heatmaps from the dataset
    heatmap_features = np.array(all_heatmaps)


def extract_features(model_name='vgg16', dataset_directory='../search_engine/resources_100k/imgs/'):
    # Define the pre-trained model and the transformation for image preprocessing
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Remove the classifier (fully connected layers) from VGG-16
        model = torch.nn.Sequential(*list(model.children())[:-1])
        feature_size = 4096
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove the classifier (fully connected layer) from ResNet-50
        model = torch.nn.Sequential(*list(model.children())[:-2])
        feature_size = 2048
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    '''elif model_name == 'vit':
        # Vision Transformer (ViT) - Example; Replace with actual ViT model loading
        model = YourViTModel(pretrained=True)
        # You need to customize the removal of fully connected layers for ViT
        # Assuming your ViT model doesn't have fully connected layers at the end
        feature_size = 768  # Adjust based on your ViT model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization
        ])'''

    # Set the model to evaluation mode (no gradient computation)
    model.eval()

    # Load the dataset using PyTorch's ImageFolder
    dataset = ImageFolder(dataset_directory, transform=transform)

    # Create a DataLoader to efficiently load and preprocess the images in batches
    batch_size = 32  # Adjust as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define a function to extract visual features from the pre-trained model
    all_features = []
    i = 1
    with torch.no_grad():
        for images, _ in dataloader:
            # Forward pass to extract features
            print(i)
            features = model(images)
            all_features.append(features)
            i += 1
    features = torch.cat(all_features)

    # Convert features to NumPy array
    features_np = features.numpy()
    n_samples, n_features, _, _ = features_np.shape
    features_reshaped = features_np.reshape(n_samples, -1)
    # np.save(model_name+'data.npy', features_reshaped)
    return features_reshaped


def tsne_vis(model_name, features_reshaped):
    # Perform t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_features = tsne.fit_transform(features_reshaped)
     # Visualize the t-SNE output
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], cmap='viridis')
    plt.xlim(-20, 20)  # Set x-axis limits from 0 to 10
    plt.ylim(-20, 20)  # Set y-axis limits from 0 to 6
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('t-SNE Visualization for '+model_name+'image embedding')
    plt.savefig(model_name + "tsne.jpg")
    plt.show()


def load_data(dir="../search_engine/resources_100k/documents.jsonl"):
    data = []
    with open(dir, "r") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append(json_obj)

    print(len(data))

    i = 1
    category=[]
    for d in data:
        product = d['product']
        #print(product["category"])  # product_category  category  query
        image_url= product["MainImage"]
        category.append(product["product_category"])
        #if image_url != product["MainImage"]:
            #print(image_url)
        # download_image(image_url, "../search_engine/resources_100k/imgs/"+d['id']+".jpg")
            #print()
        #i += 1
    a=Counter(category)
    print(a)
    print(len(a))


def download_image(url, save_path):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url, stream=True)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Open a local file for writing the binary content of the image
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"Image downloaded to: {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def kmeans(model_name='vgg16', dir='../search_engine/resources_100k/imgs/'):
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Remove the classifier (fully connected layers) from VGG-16
        model = torch.nn.Sequential(*list(model.children())[:-1])
        feature_size = 4096
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove the classifier (fully connected layer) from ResNet-50
        model = torch.nn.Sequential(*list(model.children())[:-2])
        feature_size = 2048
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'vit':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            NormalizeTo01()
        ])

        model = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Move the model to the GPU
    #model.to(device)

    # Set the model to evaluation mode (no gradient computation)
    #model.eval()


    dataset = ImageFolder(dir, transform=transform)

    # Create a DataLoader to efficiently load and preprocess the images in batches
    batch_size = 32  # Adjust as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to store feature representations
    features_list = []

    # Extract visual features from the subset of images
    with torch.no_grad():
        for images, _ in dataloader:
            # Move images to GPU
            images = images.to(device)

            # Forward pass to extract features
            features = model(images)
            print(features.data)
            d = torch.tensor(features.data['pixel_values'])
            # Convert features to NumPy arrays and flatten them
            new_f = d.view(d.size(0), -1)


            # Append features to the list
            features_list.append(new_f)

    # Concatenate the extracted features
    features_np = np.concatenate(features_list, axis=0)
    np.save(model_name + 'kmeans.npy', features_np)
    # Perform K-means clustering on the features
    num_clusters = 5  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    print(features_np.shape)
    cluster_labels = kmeans.fit_predict(features_np)

    # Visualize K-means results
    # Note: You can customize the visualization based on your requirements
    plt.scatter(features_np[:, 0], features_np[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.title('K-means Clustering Results')
    plt.show()


def sift_features(images):
    sift_vectors = {}
    descriptor_list = []  # order reserved
    sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create()
    key_num = 5
    for idx in range(len(images)):
        grey, key = images[idx]
        if key > 3:  # only use class 0-3
            continue
        img = np.array(grey)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            num = des.shape[0]
            if num < key_num:
                des = des.reshape(-1, 1)
                pad = np.zeros((128*(key_num-num), 1))
                des = np.vstack((des, pad))
            else:
                des = des[:key_num, :]
                des = des.reshape(-1, 1)
        else:
            des = np.zeros((128*key_num, 1))
        descriptor_list.extend(des)
        # print(len(descriptor_list))
        if key in sift_vectors.keys():
            sift_vectors[key].append(des)
        else:
            sift_vectors[key] = [des]
    descriptor_list = np.reshape(descriptor_list, (-1, 128 * key_num))
    # print(descriptor_list.shape)
    return [descriptor_list, sift_vectors]


def sift_kmeans():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder('../search_engine/resources_100k/imgs/', transform=transform)
    # Create a DataLoader to efficiently load and preprocess the images in batches
    batch_size = 32  # Adjust as needed
    #trainset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    descriptor_list, train_bovw_feature = sift_features(dataset)

    c_num = 20  # number of class
    kmeans = KMeans(n_clusters=c_num, init='k-means++')
    cluster_labels = kmeans.fit_predict(descriptor_list)

    plt.scatter(descriptor_list[:, 0], descriptor_list[:, 1], c=cluster_labels, s=10)
    plt.xlim(-1, 1)  # Set x-axis limits from 0 to 10
    plt.ylim(-1, 1)  # Set y-axis limits from 0 to 6
    plt.title('K-means Clustering Results')
    plt.show()


def pca_kmeans():
    # features_np = np.load("vitkmeans.npy")
    tsne_vis('vit', features_np)
    print(features_np.shape)

    # Create a PCA object with the desired number of components
    # pca = PCA(n_components=512)

    # Fit and transform your feature vectors
    # reduced_features = pca.fit_transform(features_np)
    # tsne_vis('pca',reduced_features)
    # print(reduced_features.shape)
    reduced_features = torch.load("feat_conv.pt")[:1000]
    num_clusters = 5  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Plotting the centroids
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')

    print(Counter(cluster_labels))
    # Visualize K-means results
    # Note: You can customize the visualization based on your requirements
    centroids = kmeans.cluster_centers_
    cs = ["r", "b", "c", "g", "m"]
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, s=20, alpha=0.6)

    # for i in range(len(cluster_labels)):
    #    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], c=cs[cluster_labels[i]], s=20, alpha=0.6,label='cluster '+str(cluster_labels[i]))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=30, color='k', marker='X', label='Centroids')
    # plt.xlim(-0.9935, -0.992)  # Set x-axis limits from 0 to 10
    # plt.ylim(-0.9935, -0.992)  # Set y-axis limits from 0 to 6
    plt.legend()
    plt.title('K-means Clustering for ResNet embeddings')
    plt.savefig("vit" + "pcakmeans.jpg")
    plt.show()

    # query
    # Counter({'beauty': 238, 'electronics': 229, 'garden': 216, 'fashion': 200, 'grocery': 117})
    # Counter({4: 405, 2: 204, 0: 166, 3: 153, 1: 70})'''
    '''model_name="resnet50"
    features_np = np.load(model_name + 'kmeans.npy')
    # Perform K-means clustering on the features
    num_clusters = 5  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    cluster_labels = kmeans.fit_predict(features_np)
    print(Counter(cluster_labels))
    # Visualize K-means results
    # Note: You can customize the visualization based on your requirements
    plt.scatter(features_np[:, 0], features_np[:, 1], c=cluster_labels, s=5)
    #plt.xlim(-1, 1)  # Set x-axis limits from 0 to 10
    #plt.ylim(-1, 1)  # Set y-axis limits from 0 to 6
    plt.title('K-means Clustering Results')
    plt.savefig(model_name+"kmeans.jpg")
    plt.show()'''


if __name__ == "__main__":
    f = np.load("vitkmeans.npy")
    features_np = np.load("vitkmeans.npy")
    #tsne_vis('vit', features_np)
    #print(features_np.shape)

    # Create a PCA object with the desired number of components
    pca = PCA(n_components=512)

    # Fit and transform your feature vectors
    reduced_features = pca.fit_transform(features_np)
    # tsne_vis('pca',reduced_features)
    # print(reduced_features.shape)
    #reduced_features = torch.load("feat_conv.pt")[:1000]
    num_clusters = 5  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Plotting the centroids
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')

    print(Counter(cluster_labels))
    # Visualize K-means results
    # Note: You can customize the visualization based on your requirements
    centroids = kmeans.cluster_centers_
    cs = ["r", "b", "c", "g", "m"]
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, s=20, alpha=0.6)

    # for i in range(len(cluster_labels)):
    #    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], c=cs[cluster_labels[i]], s=20, alpha=0.6,label='cluster '+str(cluster_labels[i]))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=30, color='k', marker='X', label='Centroids')
    # plt.xlim(-0.9935, -0.992)  # Set x-axis limits from 0 to 10
    # plt.ylim(-0.9935, -0.992)  # Set y-axis limits from 0 to 6
    plt.legend()
    plt.title('K-means Clustering for ResNet embeddings')
    #plt.savefig("vit" + "pcakmeans.jpg")
    plt.show()