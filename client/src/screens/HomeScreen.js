import React, { useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, Alert, Linking } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';
import { useDispatch, useSelector } from 'react-redux';
import styles from '../styles';
import { MIN_LOADING_DURATION_MS } from '../config';
import { useCheckHealthQuery, useUploadPhotoMutation } from '../services/api';
import { recordSighting } from '../store/lifeListSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';

import Header from '../components/Header';
import PlaceholderCard from '../components/PlaceholderCard';
import ImageCard from '../components/ImageCard';
import LoadingCard from '../components/LoadingCard';
import ResultCard from '../components/ResultCard';
import BirdPatternBackground from '../components/BirdPatternBackground';

// fetchBaseQuery's error shape: { status: <http code> } for a bad response,
// { status: 'FETCH_ERROR', error: <message> } for a network failure.
function describeQueryError(err, httpPrefix) {
    if (typeof err?.status === 'number') return `${httpPrefix} (${err.status})`;
    return err?.error || 'Something went wrong. Please try again.';
}

export default function HomeScreen({ onOpenLifeList, onOpenTrophies }) {
    const [imageUri, setImageUri] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const dispatch = useDispatch();
    const sightingsCount = useSelector((state) => state.lifeList.sightings.length);
    const trophyCategories = useTrophyCategories();
    const allTrophies = trophyCategories.flatMap((c) => c.trophies || []);
    const unlockedTrophyCount = allTrophies.filter((t) => t.unlocked).length;
    const totalTrophyCount = allTrophies.length;

    const { data: healthData, error: healthQueryError, isFetching: healthLoading, refetch: refetchHealth } = useCheckHealthQuery();
    const [uploadPhoto] = useUploadPhotoMutation();

    const apiHealth = healthQueryError
        ? { status: 'unhealthy', error: describeQueryError(healthQueryError, 'HTTP') }
        : healthData
            ? { status: 'healthy', ...healthData }
            : null;

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
        const startedAt = Date.now();
        try {
            const filename = uri.split('/').pop();
            const match = /(\.[0-9a-z]+)$/i.exec(filename);
            const type = match ? `image/${match[1].replace('.', '')}` : `image`;

            const formData = new FormData();
            formData.append('image', {
                uri,
                name: filename,
                type,
            });

            const json = await uploadPhoto(formData).unwrap();
            setResult(json);
        } catch (err) {
            setError(describeQueryError(err, 'Server error'));
        } finally {
            const remaining = MIN_LOADING_DURATION_MS - (Date.now() - startedAt);
            if (remaining > 0) {
                await new Promise((resolve) => setTimeout(resolve, remaining));
            }
            setLoading(false);
        }
    };

    const resetToHome = () => {
        setImageUri(null);
        setResult(null);
        setError(null);
    };

    const handleSaveSighting = (speciesId, confidence) => {
        dispatch(recordSighting({ speciesId, confidence, sourceUri: imageUri }));
    };

    return (
        <SafeAreaView style={styles.container}>
            <BirdPatternBackground />

            <Header apiHealth={apiHealth} healthLoading={healthLoading} onRetryHealth={refetchHealth} />

            <ScrollView
                contentContainerStyle={styles.scrollContainer}
                showsVerticalScrollIndicator={false}
                style={styles.mainContent}
            >
                {/* Main result card */}
                {result && !loading && <ResultCard uri={imageUri} result={result} onSave={handleSaveSighting} />}

                {/* Photo or placeholder */}
                {!result && !loading && imageUri && <ImageCard uri={imageUri} />}
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

                        <Text style={styles.sectionLabel}>Your collection</Text>

                        <TouchableOpacity style={styles.tile} onPress={onOpenTrophies} activeOpacity={0.8}>
                            <View style={styles.tileLeft}>
                                <View style={styles.tileIcon}><MaterialCommunityIcons name="trophy" size={22} color={styles.PALETTE.primary} /></View>
                                <View>
                                    <Text style={styles.tileText}>Trophy Cabinet</Text>
                                    <Text style={styles.tileSub}>Collect rarity trophies</Text>
                                </View>
                            </View>
                            <View style={styles.tileCountBadge}>
                                <Text style={styles.tileCountBadgeText}>{unlockedTrophyCount}/{totalTrophyCount}</Text>
                            </View>
                        </TouchableOpacity>

                        <TouchableOpacity style={styles.tile} onPress={onOpenLifeList} activeOpacity={0.8}>
                            <View style={styles.tileLeft}>
                                <View style={styles.tileIcon}><Feather name="book-open" size={22} color={styles.PALETTE.primary} /></View>
                                <View>
                                    <Text style={styles.tileText}>My Sightings Log</Text>
                                    <Text style={styles.tileSub}>Your recorded sightings</Text>
                                </View>
                            </View>
                            <View style={styles.tileCountBadge}>
                                <Text style={styles.tileCountBadgeText}>{sightingsCount}</Text>
                            </View>
                        </TouchableOpacity>
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

            </ScrollView>

            {/* Identify another bird / back to home — fixed to the bottom, like Header is fixed to the top */}
            {result && !loading && (
                <View style={styles.resultFooter}>
                    <TouchableOpacity style={styles.resultActionButton} onPress={pickImage}>
                        <Text style={styles.resultActionText}>Identify Another Bird</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.resultActionButtonSecondary} onPress={resetToHome}>
                        <Text style={styles.resultActionTextSecondary}>Back to Home</Text>
                    </TouchableOpacity>
                </View>
            )}

        </SafeAreaView>
    );
}
