import React, { useState, useEffect } from 'react';
import { SafeAreaView, ScrollView, StatusBar, View, Text, TouchableOpacity, Alert, Linking } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import { API_BASE } from '../config';

import Header from '../components/Header';
import PlaceholderCard from '../components/PlaceholderCard';
import ImageCard from '../components/ImageCard';
import LoadingCard from '../components/LoadingCard';
import ResultCard from '../components/ResultCard';

const API_URL = `${API_BASE}/predict`;

export default function HomeScreen() {
    const [imageUri, setImageUri] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [apiHealth, setApiHealth] = useState(null);
    const [healthLoading, setHealthLoading] = useState(false);

    const pickImage = async () => {
        const permission = await ImagePicker.requestCameraPermissionsAsync();
        if (permission.status !== 'granted') {
            if (permission.canAskAgain === false) {
                Alert.alert(
                    'Camera access needed',
                    'Camera access is currently blocked. Enable it in Settings to identify birds by photo.',
                    [
                        { text: 'Cancel', style: 'cancel' },
                        { text: 'Open Settings', onPress: () => Linking.openSettings() },
                    ]
                );
            } else {
                Alert.alert('Camera permission required', 'We need camera access to take a photo of the bird.');
            }
            return;
        }
        let res = await ImagePicker.launchCameraAsync({
            base64: false,
            quality: 0.8,
        });

        if (!res.canceled) {
            const uri = res.assets[0].uri;
            setImageUri(uri);
            uploadImage(uri);
        }
    };

    const uploadImage = async (uri) => {
        setLoading(true);
        setResult(null);
        setError(null);
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

            if (!res.ok) {
                throw new Error(`Server error (${res.status})`);
            }

            const json = await res.json();
            setResult(json);
        } catch (err) {
            setError(err.message || 'Something went wrong. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const resetToHome = () => {
        setImageUri(null);
        setResult(null);
        setError(null);
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

    useEffect(() => {
        checkApiHealth();
    }, []);

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content"
                backgroundColor={styles.header.backgroundColor}
                showHideTransition={'slide'}
                networkActivityIndicatorVisible={true} />

            <Header apiHealth={apiHealth} healthLoading={healthLoading} onRetryHealth={checkApiHealth} />

            <ScrollView
                contentContainerStyle={styles.scrollContainer}
                showsVerticalScrollIndicator={false}
                style={styles.mainContent}
            >
                {/* Main result card */}
                {result && !loading && <ResultCard uri={imageUri} result={result} />}

                {/* Photo or placeholder */}
                {!result && imageUri && <ImageCard uri={imageUri} />}
                {!result && !imageUri && (
                    <>
                        <TouchableOpacity style={styles.heroCard} onPress={pickImage} activeOpacity={0.85}>
                            <View style={styles.heroIconCircle}>
                                <Feather name="camera" size={32} color="#ffffff" />
                            </View>
                            <Text style={styles.heroTitle}>Identify a Bird</Text>
                            <Text style={styles.heroSubtitle}>Point your camera at a bird and we'll tell you what it is.</Text>
                            <View style={styles.heroButton}>
                                <Text style={styles.heroButtonText}>Take a Photo</Text>
                            </View>
                        </TouchableOpacity>

                        <Text style={styles.sectionLabel}>Coming soon</Text>

                        <View style={styles.tileDisabled}>
                            <View style={styles.tileLeft}>
                                <View style={styles.tileIconDisabled}><MaterialCommunityIcons name="bird" size={22} color={styles.PALETTE.mutedText} /></View>
                                <View>
                                    <Text style={styles.tileTextDisabled}>Browse Species</Text>
                                    <Text style={styles.tileSub}>Explore species reference</Text>
                                </View>
                            </View>
                            <View style={styles.soonBadge}><Text style={styles.soonBadgeText}>Soon</Text></View>
                        </View>

                        <View style={styles.tileDisabled}>
                            <View style={styles.tileLeft}>
                                <View style={styles.tileIconDisabled}><Feather name="book-open" size={22} color={styles.PALETTE.mutedText} /></View>
                                <View>
                                    <Text style={styles.tileTextDisabled}>My Sightings Log</Text>
                                    <Text style={styles.tileSub}>Your recorded sightings</Text>
                                </View>
                            </View>
                            <View style={styles.soonBadge}><Text style={styles.soonBadgeText}>Soon</Text></View>
                        </View>
                    </>
                )}

                {/* Error state */}
                {error && !loading && (
                    <View style={styles.errorCard}>
                        <Text style={styles.errorTitle}>Couldn't identify that photo</Text>
                        <Text style={styles.errorText}>{error}</Text>
                        <View style={styles.errorButtonRow}>
                            <TouchableOpacity style={styles.resultActionButton} onPress={() => uploadImage(imageUri)}>
                                <Text style={styles.resultActionText}>Retry</Text>
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.resultActionButtonSecondary} onPress={pickImage}>
                                <Text style={styles.resultActionTextSecondary}>New Photo</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                )}

                {/* Loading state */}
                {loading && <LoadingCard />}

                {/* Identify another bird / back to home */}
                {result && !loading && (
                    <View style={[styles.errorButtonRow, styles.newPhotoButton]}>
                        <TouchableOpacity style={styles.resultActionButton} onPress={pickImage}>
                            <Text style={styles.resultActionText}>Identify Another Bird</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.resultActionButtonSecondary} onPress={resetToHome}>
                            <Text style={styles.resultActionTextSecondary}>Back to Home</Text>
                        </TouchableOpacity>
                    </View>
                )}

            </ScrollView>

        </SafeAreaView>
    );
}
