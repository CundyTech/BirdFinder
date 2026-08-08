import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { API_BASE, API_KEY } from '../config';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE,
    prepareHeaders: (headers) => {
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
