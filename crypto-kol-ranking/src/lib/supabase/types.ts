export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      tokens: {
        Row: {
          id: number;
          symbol: string;
          score: number | null;
          mentions: number;
          unique_kols: number;
          sentiment: number | null;
          momentum: number | null;
          breadth: number | null;
          conviction_weighted: number | null;
          trend: "up" | "down" | "stable" | null;
          time_window: "3h" | "6h" | "12h" | "24h" | "48h" | "7d";
          change_24h: number | null;
          change_7d: number | null;
          updated_at: string;
        };
        Insert: {
          id?: number;
          symbol: string;
          score?: number | null;
          mentions?: number;
          unique_kols?: number;
          sentiment?: number | null;
          momentum?: number | null;
          breadth?: number | null;
          conviction_weighted?: number | null;
          trend?: "up" | "down" | "stable" | null;
          time_window: "3h" | "6h" | "12h" | "24h" | "48h" | "7d";
          change_24h?: number | null;
          change_7d?: number | null;
          updated_at?: string;
        };
        Update: {
          id?: number;
          symbol?: string;
          score?: number | null;
          mentions?: number;
          unique_kols?: number;
          sentiment?: number | null;
          momentum?: number | null;
          breadth?: number | null;
          conviction_weighted?: number | null;
          trend?: "up" | "down" | "stable" | null;
          time_window?: "3h" | "6h" | "12h" | "24h" | "48h" | "7d";
          change_24h?: number | null;
          change_7d?: number | null;
          updated_at?: string;
        };
      };
      groups: {
        Row: {
          id: number;
          name: string;
          telegram_id: number | null;
          telegram_username: string | null;
          conviction: number | null;
          category: string | null;
          active: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: number;
          name: string;
          telegram_id?: number | null;
          telegram_username?: string | null;
          conviction?: number | null;
          category?: string | null;
          active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: number;
          name?: string;
          telegram_id?: number | null;
          telegram_username?: string | null;
          conviction?: number | null;
          category?: string | null;
          active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
      };
      mentions: {
        Row: {
          id: number;
          symbol: string;
          group_id: number | null;
          message_id: number | null;
          message_text: string | null;
          sentiment: number | null;
          created_at: string;
        };
        Insert: {
          id?: number;
          symbol: string;
          group_id?: number | null;
          message_id?: number | null;
          message_text?: string | null;
          sentiment?: number | null;
          created_at?: string;
        };
        Update: {
          id?: number;
          symbol?: string;
          group_id?: number | null;
          message_id?: number | null;
          message_text?: string | null;
          sentiment?: number | null;
          created_at?: string;
        };
      };
      profiles: {
        Row: {
          id: string;
          email: string | null;
          plan: "free" | "pro" | "enterprise";
          stripe_customer_id: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          email?: string | null;
          plan?: "free" | "pro" | "enterprise";
          stripe_customer_id?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string | null;
          plan?: "free" | "pro" | "enterprise";
          stripe_customer_id?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      subscriptions: {
        Row: {
          id: number;
          user_id: string | null;
          stripe_subscription_id: string | null;
          stripe_price_id: string | null;
          status: "active" | "canceled" | "past_due" | "incomplete" | "trialing" | null;
          current_period_start: string | null;
          current_period_end: string | null;
          cancel_at_period_end: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: number;
          user_id?: string | null;
          stripe_subscription_id?: string | null;
          stripe_price_id?: string | null;
          status?: "active" | "canceled" | "past_due" | "incomplete" | "trialing" | null;
          current_period_start?: string | null;
          current_period_end?: string | null;
          cancel_at_period_end?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: number;
          user_id?: string | null;
          stripe_subscription_id?: string | null;
          stripe_price_id?: string | null;
          status?: "active" | "canceled" | "past_due" | "incomplete" | "trialing" | null;
          current_period_start?: string | null;
          current_period_end?: string | null;
          cancel_at_period_end?: boolean;
          created_at?: string;
          updated_at?: string;
        };
      };
      api_keys: {
        Row: {
          id: number;
          user_id: string | null;
          key_hash: string;
          name: string | null;
          last_used_at: string | null;
          requests_today: number;
          created_at: string;
        };
        Insert: {
          id?: number;
          user_id?: string | null;
          key_hash: string;
          name?: string | null;
          last_used_at?: string | null;
          requests_today?: number;
          created_at?: string;
        };
        Update: {
          id?: number;
          user_id?: string | null;
          key_hash?: string;
          name?: string | null;
          last_used_at?: string | null;
          requests_today?: number;
          created_at?: string;
        };
      };
      scrape_metadata: {
        Row: {
          id: number;
          updated_at: string;
          stats: Json;
        };
        Insert: {
          id?: number;
          updated_at?: string;
          stats?: Record<string, unknown>;
        };
        Update: {
          id?: number;
          updated_at?: string;
          stats?: Record<string, unknown>;
        };
      };
    };
    Functions: {
      get_token_ranking: {
        Args: {
          p_time_window: string;
          p_limit: number;
          p_offset: number;
        };
        Returns: {
          rank: number;
          symbol: string;
          score: number;
          mentions: number;
          unique_kols: number;
          sentiment: number;
          trend: string;
          change_24h: number;
        }[];
      };
    };
    Enums: {};
  };
}
