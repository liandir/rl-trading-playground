"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useRegistry() {
  return useQuery({
    queryKey: ["registry"],
    queryFn: api.registry,
    staleTime: 60 * 60 * 1000,
  });
}
