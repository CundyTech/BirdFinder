import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { API_BASE, API_KEY } from '../config';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE,
    prepareHeaders: (headers) => {
      if (!API_KEY) {
        console.warn('EXPO_PUBLIC_API_KEY missing — X-API-Key will not be sent')
      } else {
        // Mask the key in logs: show first 6 and last 4 chars plus length
        const len = API_KEY.length
        const masked = API_KEY.length > 10 ? `${API_KEY.slice(0,6)}…${API_KEY.slice(-4)}` : API_KEY
        console.log(`Setting X-API-Key (masked=${masked} len=${len})`)
      }
      headers.set('X-API-Key', API_KEY);
      return headers;
    },
  }),
  endpoints: (builder) => ({
    checkHealth: builder.query({
      query: () => 'health',
    }),
    uploadPhoto: builder.mutation({
      query: (formData) => ({
        url: 'predict',
        method: 'POST',
        body: formData,
      }),
    }),
  }),
});

export const { useCheckHealthQuery, useUploadPhotoMutation } = api;
