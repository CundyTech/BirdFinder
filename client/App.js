import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Dimensions
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const { width } = Dimensions.get('window');
const API_URL = 'http://192.168.50.160:8080/predict'; // change to your machine IP or localhost as appropriate

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);

  const pickImage = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (permission.status !== 'granted') {
      alert('Camera permission required');
      return;
    }
    let result = await ImagePicker.launchCameraAsync({
      base64: false,
      quality: 0.8,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setImageUri(uri);
      uploadImage(uri);
    }
  };

  const uploadImage = async (uri) => {
    setLoading(true);
    setResult(null);
    try {
      const localUri = uri;
      const filename = localUri.split('/').pop();
      const match = /(\.[0-9a-z]+)$/i.exec(filename);
      const type = match ? `image/${match[1].replace('.', '')}` : `image`;

      const formData = new FormData();
      formData.append('image', {
        uri: localUri,
        name: filename,
        type,
      });

      const res = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const json = await res.json();
      setResult(json);
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const checkApiHealth = async () => {
    setHealthLoading(true);
    try {
      const healthUrl = API_URL.replace('/predict', '/health');
      const res = await fetch(healthUrl, {
        method: 'GET',
      });

      if (res.ok) {
        const json = await res.json();
        setApiHealth({ status: 'healthy', ...json });
      } else {
        setApiHealth({ status: 'unhealthy', error: `HTTP ${res.status}` });
      }
    } catch (err) {
      setApiHealth({ status: 'unhealthy', error: err.message });
    } finally {
      setHealthLoading(false);
    }
  };

  // Check API health on app start
  useEffect(() => {
    checkApiHealth();
  }, []);

  const getConfidencePercentage = () => {
    if (!result || !result.scores) return 0;
    const maxScore = Math.max(...result.scores);
    return (maxScore * 100).toFixed(1);
  };

  const formatBirdName = (className) => {
    if (!className) return '';
    // Remove the number prefix and convert underscores to spaces
    const name = className.replace(/^\d+\./, '').replace(/_/g, ' ');
    return name;
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={styles.header.backgroundColor} />

      {/* Main Content */}
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerTop}>
            <Text style={styles.title}>🦅 BirdFinder</Text>
            <View style={styles.headerTag}>
              <Text style={styles.headerTagText}>UK Edition</Text>
            </View>
          </View>
          <View style={styles.headerBottom}>
            <Text style={styles.subtitle}>Identify a bird with one photo</Text>
            {healthLoading ? (
              <ActivityIndicator size="small" color={styles.primary.color} />
            ) : (
              <View style={[
                styles.statusIndicator,
                apiHealth?.status === 'healthy' ? styles.statusHealthy : styles.statusUnhealthy
              ]} />
            )}
          </View>
        </View>

        {/* Initial Placeholder */}
        {!imageUri && !result && (
          <View style={styles.placeholderContainer}>
            <View style={styles.placeholderCard}>
              <Text style={styles.placeholderIcon}>📷</Text>
              <Text style={styles.placeholderTitle}>Ready to Identify Birds?</Text>
              <Text style={styles.placeholderText}>
                Take a photo of any bird and our AI will identify it for you.
                Works best with clear, well-lit photos of UK birds.
              </Text>
            </View>
          </View>
        )}

        {/* Image Display */}
        {imageUri && !result && (
          <View style={styles.imageCard}>
            <Image source={{ uri: imageUri }} style={styles.image} />
          </View>
        )}

        {/* Loading State */}
        {loading && (
          <View style={styles.loadingCard}>
            <ActivityIndicator size="large" color={styles.primary.color} />
            <Text style={styles.loadingText}>Analyzing image...</Text>
            <Text style={styles.loadingSubtext}>This may take a few seconds</Text>
          </View>
        )}

        {/* Results */}
        {result && !loading && (
          <View style={styles.resultCard}>
            {/* Merged Image and Results */}
            {imageUri && (
              <View style={styles.resultImageContainer}>
                <Image source={{ uri: imageUri }} style={styles.resultImage} />
              </View>
            )}

            <View style={styles.resultHeader}>
              <Text style={styles.resultTitle}>🎯 Identification Result</Text>
              <View style={styles.confidenceBadge}>
                <Text style={styles.confidenceText}>{getConfidencePercentage()}%</Text>
              </View>
            </View>

            <View style={styles.birdInfo}>
              <Text style={styles.birdName}>{formatBirdName(result.predicted_class)}</Text>
              <Text style={styles.birdClass}>{result.predicted_class}</Text>
            </View>

            <View style={styles.modelInfo}>
              <Text style={styles.modelLabel}>AI Model:</Text>
              <Text style={styles.modelName}>
                {result.model_path.includes('smoke') ? 'Demo Model' : 'Production Model'}
              </Text>
            </View>
          </View>
        )}

      </ScrollView>

      {/* Bottom Action Panel */}
      <View style={styles.actionPanel}>
        <TouchableOpacity style={styles.mainActionButton} onPress={pickImage}>
          <Text style={styles.mainActionText}>📸 Take Photo</Text>
          <Text style={styles.mainActionSubtext}>Identify a bird</Text>
        </TouchableOpacity>
      </View>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  mainContent: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingBottom: 120, // Extra padding for the action panel
  },
  header: {
    backgroundColor: '#1a1a1a',
    paddingVertical: 30,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 25,
    borderBottomRightRadius: 25,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginRight: 10,
  },
  subtitle: {
    fontSize: 15,
    color: '#c2c9d6',
    textAlign: 'center',
    opacity: 0.85,
  },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  headerBottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  headerTag: {
    backgroundColor: '#274571',
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 4,
    marginLeft: 10,
  },
  headerTagText: {
    color: '#d6e4ff',
    fontSize: 12,
    fontWeight: '700',
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginLeft: 10,
  },
  statusHealthy: {
    backgroundColor: '#4CAF50',
  },
  statusUnhealthy: {
    backgroundColor: '#f44336',
  },
  imageCard: {
    backgroundColor: '#141f33',
    borderRadius: 20,
    padding: 18,
    marginBottom: 18,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.22,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 1,
    borderColor: '#1f2f4d',
  },
  image: {
    width: width - 70,
    height: width - 70,
    borderRadius: 18,
    resizeMode: 'cover',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  placeholderCard: {
    backgroundColor: '#141f33',
    borderRadius: 22,
    padding: 28,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 10,
    borderWidth: 1,
    borderColor: '#1c2a45',
    maxWidth: width - 40,
  },
  placeholderIcon: {
    fontSize: 72,
    marginBottom: 18,
    opacity: 0.85,
  },
  placeholderTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 12,
  },
  placeholderText: {
    fontSize: 15,
    color: '#b9c6dc',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 22,
  },
  placeholderHint: {
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#444',
  },
  placeholderHintText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
  loadingCard: {
    backgroundColor: '#1e1e1e',
    borderRadius: 16,
    padding: 30,
    marginBottom: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 6,
    borderWidth: 1,
    borderColor: '#333',
  },
  loadingText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 15,
  },
  loadingSubtext: {
    color: '#b0b0b0',
    fontSize: 14,
    marginTop: 5,
  },
  resultCard: {
    backgroundColor: '#141f33',
    borderRadius: 20,
    padding: 22,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.22,
    shadowRadius: 14,
    elevation: 10,
    borderWidth: 1,
    borderColor: '#22416d',
  },
  resultImageContainer: {
    marginBottom: 22,
    alignItems: 'center',
  },
  resultImage: {
    width: width - 80,
    height: width - 80,
    borderRadius: 20,
    resizeMode: 'cover',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  confidenceBadge: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  confidenceText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  birdInfo: {
    marginBottom: 20,
  },
  birdName: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 5,
  },
  birdClass: {
    fontSize: 16,
    color: '#b0b0b0',
    fontStyle: 'italic',
  },
  modelInfo: {
    backgroundColor: '#17223b',
    padding: 16,
    borderRadius: 14,
    marginBottom: 20,
  },
  modelLabel: {
    fontSize: 13,
    color: '#9fb2d1',
    marginBottom: 6,
    letterSpacing: 0.4,
  },
  modelName: {
    fontSize: 16,
    color: '#e6f1ff',
    fontWeight: '600',
  },
  // Bottom Action Panel Styles
  actionPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#1a1a1a',
    paddingHorizontal: 20,
    paddingVertical: 25,
    paddingBottom: 35, // Extra padding for safe area
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  mainActionButton: {
    backgroundColor: '#2196F3',
    paddingVertical: 20,
    paddingHorizontal: 40,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#2196F3',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 12,
  },
  mainActionText: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  mainActionSubtext: {
    color: '#e3f2fd',
    fontSize: 14,
    marginTop: 4,
    opacity: 0.9,
  },
  primary: {
    color: '#2196F3',
  },
});
