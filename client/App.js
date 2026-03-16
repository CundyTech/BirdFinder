import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = 'http://10.0.2.2:8080/predict'; // change to your machine IP or localhost as appropriate

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BirdFinder</Text>
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
