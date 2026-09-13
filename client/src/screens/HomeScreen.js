import React, { useEffect, useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, Alert, Linking } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';
import { useDispatch, useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { MIN_LOADING_DURATION_MS } from '../config';
import i18n from '../i18n';
import { useCheckHealthQuery, useUploadPhotoMutation } from '../services/api';
import { recordSighting } from '../store/lifeListSlice';
import { spendFilm } from '../store/filmSlice';
import { recordActivity } from '../store/streakSlice';
import { spendReroll } from '../store/rewardsSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';

import Header from '../components/Header';
import PlaceholderCard from '../components/PlaceholderCard';
import ImageCard from '../components/ImageCard';
import LoadingCard from '../components/LoadingCard';
import ResultCard from '../components/ResultCard';
import BirdPatternBackground from '../components/BirdPatternBackground';
import OutOfFilmModal from '../components/OutOfFilmModal';
import FilmChoiceModal from '../components/FilmChoiceModal';

// fetchBaseQuery's error shape: { status: <http code> } for a bad response,
// { status: 'FETCH_ERROR', error: <message> } for a network failure.
function describeQueryError(err, httpPrefix) {
    if (typeof err?.status === 'number') return `${httpPrefix} (${err.status})`;
    return err?.error || i18n.t('home.genericError');
}

export default function HomeScreen({ onOpenLifeList, onOpenTrophies }) {
    const { t } = useTranslation();
    const [imageUri, setImageUri] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showOutOfFilm, setShowOutOfFilm] = useState(false);
    const [showFilmChoice, setShowFilmChoice] = useState(false);
    // Set right before openCamera when an ad (from FilmChoiceModal) earns
    // this identification for free — read once, by uploadImage's success
    // handler, then cleared in its finally so it never leaks into a later,
    // unrelated attempt.
    const [freeViaAd, setFreeViaAd] = useState(false);

    const dispatch = useDispatch();
    const sightingsCount = useSelector((state) => state.lifeList.sightings.length);
    const filmBalance = useSelector((state) => state.film.balance);
    const rerollTokens = useSelector((state) => state.rewards.rerollTokens);
    const unlockedForever = useSelector((state) => state.premium.unlockedForever);
    const trophyCategories = useTrophyCategories();
    const allTrophies = trophyCategories.flatMap((c) => c.trophies || []);
    const unlockedTrophyCount = allTrophies.filter((trophy) => trophy.unlocked).length;
    const totalTrophyCount = allTrophies.length;

    // What this identification will actually cost — mirrors the spend
    // decision in uploadImage's success handler below (an ad-earned free
    // pass first, then Film, then a reroll token only once Film is at
    // zero) so the loading screen never shows a different answer than what
    // actually gets charged.
    const spendKind = unlockedForever
        ? 'unlimited'
        : freeViaAd
            ? 'free-ad'
            : filmBalance > 0
                ? 'film'
                : 'reroll';

    useEffect(() => {
        if (showOutOfFilm && (unlockedForever || filmBalance > 0)) {
            setShowOutOfFilm(false);
        }
    }, [showOutOfFilm, unlockedForever, filmBalance]);

    const { data: healthData, error: healthQueryError, isFetching: healthLoading, refetch: refetchHealth } = useCheckHealthQuery();
    const [uploadPhoto] = useUploadPhotoMutation();

    const apiHealth = healthQueryError
        ? { status: 'unhealthy', error: describeQueryError(healthQueryError, t('home.healthErrorPrefix')) }
        : healthData
            ? { status: 'healthy', ...healthData }
            : null;

    const openCamera = async () => {
        const permission = await ImagePicker.requestCameraPermissionsAsync();
        if (permission.status !== 'granted') {
            if (permission.canAskAgain === false) {
                Alert.alert(
                    t('home.cameraAccessNeededTitle'),
                    t('home.cameraAccessNeededMessage'),
                    [
                        { text: t('common.cancel'), style: 'cancel' },
                        { text: t('home.openSettings'), onPress: () => Linking.openSettings() },
                    ]
                );
            } else {
                Alert.alert(t('home.cameraPermissionRequiredTitle'), t('home.cameraPermissionRequiredMessage'));
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

    // Nothing is ever spent silently — unlocked-forever players go
    // straight to the camera, everyone else always sees a choice first:
    // FilmChoiceModal (Film vs. an ad) while Film is available, or
    // OutOfFilmModal (reroll/ad/unlock) once it's at zero.
    //
    // freeViaAd resets here, at the one entry point for starting a new
    // attempt, rather than only after a successful upload — otherwise a
    // free pass earned but never used (e.g. the camera got cancelled
    // before a photo was taken) would silently carry over and cover a
    // later attempt the user explicitly chose to pay Film or a reroll for.
    const pickImage = async () => {
        setFreeViaAd(false);
        if (unlockedForever) {
            await openCamera();
            return;
        }
        if (filmBalance <= 0) {
            setShowOutOfFilm(true);
            return;
        }
        setShowFilmChoice(true);
    };

    const handleUseReroll = async () => {
        setShowOutOfFilm(false);
        await openCamera();
    };

    const handleUseFilm = async () => {
        setShowFilmChoice(false);
        await openCamera();
    };

    // FilmChoiceModal's own onClose already ran before showAd() fired, so
    // by the time the ad is actually watched and this callback fires, the
    // modal is already gone — just record the free pass and go straight to
    // the camera, same as choosing "Use 1 Film" would have.
    const handleAdEarnedFreeId = async () => {
        setFreeViaAd(true);
        await openCamera();
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
            // Only a successful identification costs anything — a failed
            // request (network/server error) didn't actually use the
            // service, so it shouldn't cost the user anything. 'unlimited'
            // and 'free-ad' cost nothing either way.
            if (spendKind === 'film') dispatch(spendFilm());
            else if (spendKind === 'reroll') dispatch(spendReroll());
            setResult(json);
        } catch (err) {
            setError(describeQueryError(err, t('home.uploadErrorPrefix')));
        } finally {
            const remaining = MIN_LOADING_DURATION_MS - (Date.now() - startedAt);
            if (remaining > 0) {
                await new Promise((resolve) => setTimeout(resolve, remaining));
            }
            setLoading(false);
            setFreeViaAd(false);
        }
    };

    const resetToHome = () => {
        setImageUri(null);
        setResult(null);
        setError(null);
    };

    const handleSaveSighting = (speciesId, confidence) => {
        dispatch(recordSighting({ speciesId, confidence, sourceUri: imageUri }));
        dispatch(recordActivity());
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
                            <Text style={styles.heroTitle}>{t('home.heroTitle')}</Text>
                            <Text style={styles.heroSubtitle}>{t('home.heroSubtitle')}</Text>
                            <View style={styles.heroButton}>
                                <Text style={styles.heroButtonText}>{t('home.heroButtonText')}</Text>
                            </View>
                        </TouchableOpacity>

                        <Text style={styles.sectionLabel}>{t('home.sectionLabel')}</Text>

                        <TouchableOpacity style={styles.tile} onPress={onOpenTrophies} activeOpacity={0.8}>
                            <View style={styles.tileLeft}>
                                <View style={styles.tileIcon}><MaterialCommunityIcons name="trophy" size={22} color={styles.PALETTE.primary} /></View>
                                <View>
                                    <Text style={styles.tileText}>{t('home.trophyCabinetTileTitle')}</Text>
                                    <Text style={styles.tileSub}>{t('home.trophyCabinetTileSub')}</Text>
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
                                    <Text style={styles.tileText}>{t('home.lifeListTileTitle')}</Text>
                                    <Text style={styles.tileSub}>{t('home.lifeListTileSub')}</Text>
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
                        <Text style={styles.errorTitle}>{t('home.errorTitle')}</Text>
                        <Text style={styles.errorText}>{error}</Text>
                        <View style={styles.errorButtonRow}>
                            <TouchableOpacity style={styles.resultActionButton} onPress={() => uploadImage(imageUri)}>
                                <Text style={styles.resultActionText}>{t('common.retry')}</Text>
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.resultActionButtonSecondary} onPress={pickImage}>
                                <Text style={styles.resultActionTextSecondary}>{t('home.newPhoto')}</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                )}

                {/* Loading state */}
                {loading && <LoadingCard spendKind={spendKind} />}

            </ScrollView>

            {/* Identify another bird / back to home — fixed to the bottom, like Header is fixed to the top */}
            {result && !loading && (
                <View style={styles.resultFooter}>
                    <TouchableOpacity style={styles.resultActionButton} onPress={pickImage}>
                        <Text style={styles.resultActionText}>{t('home.identifyAnotherBird')}</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.resultActionButtonSecondary} onPress={resetToHome}>
                        <Text style={styles.resultActionTextSecondary}>{t('home.backToHome')}</Text>
                    </TouchableOpacity>
                </View>
            )}

            <OutOfFilmModal
                visible={showOutOfFilm}
                onClose={() => setShowOutOfFilm(false)}
                rerollTokens={rerollTokens}
                onUseReroll={handleUseReroll}
            />

            <FilmChoiceModal
                visible={showFilmChoice}
                onClose={() => setShowFilmChoice(false)}
                filmBalance={filmBalance}
                onUseFilm={handleUseFilm}
                onAdEarned={handleAdEarnedFreeId}
            />

        </SafeAreaView>
    );
}
