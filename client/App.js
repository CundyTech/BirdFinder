import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

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

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BirdFinder</Text>
      
      {/* API Health Status */}
      <View style={styles.healthContainer}>
        <Text style={styles.healthTitle}>API Status:</Text>
        {healthLoading ? (
          <ActivityIndicator size="small" />
        ) : apiHealth ? (
          <Text style={[
            styles.healthStatus,
            apiHealth.status === 'healthy' ? styles.healthy : styles.unhealthy
          ]}>
            {apiHealth.status === 'healthy' ? '✅ Connected' : `❌ ${apiHealth.error || 'Disconnected'}`}
          </Text>
        ) : (
          <Text style={styles.healthStatus}>Checking...</Text>
        )}
        <Button title="Check API" onPress={checkApiHealth} />
      </View>

      <Button title="Take Photo" onPress={pickImage} />
      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
      {loading && <ActivityIndicator size="large" />}
      {result && (
        <View style={styles.result}>
          <Text>Predicted: {result.predicted_class}</Text>
          <Text>Model: {result.model_path}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    marginBottom: 12,
  },
  healthContainer: {
    alignItems: 'center',
    marginBottom: 20,
    padding: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    width: '100%',
  },
  healthTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  healthStatus: {
    fontSize: 14,
    marginBottom: 10,
  },
  healthy: {
    color: 'green',
  },
  unhealthy: {
    color: 'red',
  },
  image: {
    width: 300,
    height: 300,
    marginTop: 12,
    marginBottom: 12,
    resizeMode: 'contain',
  },
  result: {
    marginTop: 12,
  },
});
