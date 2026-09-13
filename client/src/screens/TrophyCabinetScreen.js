import React, { useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import useTrophyCategories from '../hooks/useTrophyCategories';
import TrophyCard from '../components/TrophyCard';

export default function TrophyCabinetScreen({ onBack, onOpenSpecies }) {
  const { t } = useTranslation();
  const categories = useTrophyCategories();
  const [expandedKey, setExpandedKey] = useState(null);

  const allTrophies = categories.flatMap((c) => c.trophies || []);
  const unlockedCount = allTrophies.filter((trophy) => trophy.unlocked).length;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.subHeader}>
        <TouchableOpacity
          style={styles.subHeaderBackButton}
          onPress={onBack}
          accessibilityRole="button"
          accessibilityLabel={t('common.back')}
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.textOnDark} />
        </TouchableOpacity>
        <View style={styles.subHeaderTitleWrap}>
          <Text style={styles.subHeaderTitle}>{t('trophyCabinet.title')}</Text>
          <Text style={styles.subHeaderSubtitle}>
            {allTrophies.length > 0
              ? t('trophyCabinet.subtitleProgress', { unlocked: unlockedCount, total: allTrophies.length })
              : t('trophyCabinet.subtitleLoading')}
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        {categories.map((category) => {
          const categoryUnlocked = category.trophies ? category.trophies.filter((trophy) => trophy.unlocked).length : 0;
          return (
            <View key={category.id} style={styles.trophyCategorySection}>
              <View style={styles.trophyCategoryHeader}>
                <Text style={styles.trophyCategoryTitle}>{category.label}</Text>
                {category.trophies && (
                  <Text style={styles.trophyCategoryProgress}>
                    {categoryUnlocked} / {category.trophies.length}
                  </Text>
                )}
              </View>
              <Text style={styles.trophyCategoryDescription}>{category.description}</Text>

              {!category.trophies ? (
                <View style={styles.referenceLoading}>
                  <ActivityIndicator size="small" color={styles.PALETTE.primary} />
                  <Text style={styles.referenceLoadingText}>{t('trophyCabinet.referenceLoadingText')}</Text>
                </View>
              ) : (
                category.trophies.map((trophy) => {
                  const key = `${category.id}:${trophy.key}`;
                  return (
                    <TrophyCard
                      key={key}
                      trophy={trophy}
                      expanded={expandedKey === key}
                      onToggleExpand={() => setExpandedKey((current) => (current === key ? null : key))}
                      onOpenSpecies={onOpenSpecies}
                    />
                  );
                })
              )}
            </View>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}
