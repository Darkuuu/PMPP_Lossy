// Copyright (c) 2016 Stefan Guthe, Maximilian von Buelow / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

// this in an implementation of an arithmetic coder used for the DDV4 mode

#include <math.h>
#include <vector>
#include <deque>

#define assert(C) do { if (!(C)) { std::cerr << "Assertion \"" << #C << "\" failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)
#define assert_lt(A, B) do { if (!((A) < (B))) { std::cerr << "Assertion \"" << #A << "\" (" << (A) << ") < \"" << #B << "\" (" << (B) << ") failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)
#define assert_le(A, B) do { if (!((A) <= (B))) { std::cerr << "Assertion \"" << #A << "\" (" << (A) << ") <= \"" << #B << "\" (" << (B) << ") failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)
#define assert_gt(A, B) do { if (!((A) > (B))) { std::cerr << "Assertion \"" << #A << "\" (" << (A) << ") > \"" << #B << "\" (" << (B) << ") failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)
#define assert_ge(A, B) do { if (!((A) >= (B))) { std::cerr << "Assertion \"" << #A << "\" (" << (A) << ") >= \"" << #B << "\" (" << (B) << ") failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)
#define assert_eq(A, B) do { if (!((A) == (B))) { std::cerr << "Assertion \"" << #A << "\" (" << (A) << ") == \"" << #B << "\" (" << (B) << ") failed on line " << __LINE__ << std::endl; std::exit(1); } } while(0)

#ifndef WIN32
#define __forceinline inline __attribute__((always_inline))
#endif

template <typename TF>
struct Coder {
	static const int b = sizeof(TF) * 8;
	static const int f = b - 2;
	static const TF HALF = TF(1) << (b - 1);
	static const TF QUARTER = TF(1) << (b - 2);
};

template <typename TF, typename TBO = uint64_t>
struct Encoder : Coder<TF> {
	using Coder<TF>::b;
	using Coder<TF>::f;
	using Coder<TF>::HALF;
	using Coder<TF>::QUARTER;

	TF L; // low
	TF R; // range
	TBO bits_outstanding;
	std::deque<unsigned char> &os;
	int pos = 0;
	unsigned char out;

	Encoder(std::deque<unsigned char> &_os) : os(_os)
	{
		pos = 0;
		out = 0;
	}

	void start_encode()
	{
		L = 0;
		R = HALF;
		bits_outstanding = 0;
	}

	void finish_encode()
	{
		for (int i = b - 1; i >= 0; --i) {
			bit_plus_follow((L >> i) & 1);
		}
		if (pos != 0)
		{
			os.push_back(out);
			out = 0;
			pos = 0;
		}
	}

	void __forceinline write_one_bit(unsigned char bit)
	{
		out |= bit << (7 - pos);
		pos++;
		if (pos == 8)
		{
			os.push_back(out);
			out = 0;
			pos = 0;
		}
	}

	void arithmetic_encode(TF l, TF h, TF t)
	{
		TF r = R / t;

		L = L + r * l;
		if (h < t)
			R = r * (h - l);
		else
			R = R - r * l;

		while (R <= QUARTER) {
			if (L <= HALF && L + R <= HALF) {
				bit_plus_follow(0);
			}
			else if (L >= HALF) {
				bit_plus_follow(1);
				L -= HALF;
			}
			else {
				++bits_outstanding;
				L -= QUARTER;
			}
			L *= 2;
			R *= 2;
		}
	}

	void __forceinline bit_plus_follow(unsigned char x)
	{
		write_one_bit(x);
		while (bits_outstanding > 0) {
			write_one_bit(x ^ 1);
			--bits_outstanding;
		}
	}
};

template <typename TF>
struct Decoder : Coder<TF> {
	using Coder<TF>::b;
	using Coder<TF>::f;
	using Coder<TF>::HALF;
	using Coder<TF>::QUARTER;

	TF R; // range
	TF D;
	TF r;
	std::deque<unsigned char> &is;
	int pos;

	Decoder(std::deque<unsigned char> &_is) : is(_is)
	{
		pos = 0;
	}

	void start_decode()
	{
		R = HALF;
		D = read_n_bits(b);
	}
	void finish_decode()
	{
		/* no action required */
	}

	TF __forceinline read_one_bit()
	{
		TF r = (is.front() >> (7 - pos)) & TF(1);
		pos++;
		if (pos == 8)
		{
			is.pop_front();
			pos = 0;
		}
		return r;
	}

	TF __forceinline read_n_bits(int n)
	{
		int done = 0;
		TF r = TF(is.front() & (0xff >> pos));
		done += 8 - pos;
		pos = (pos + n) & 7;
		if (done <= n)
		{
			is.pop_front();
		}
		else r >>= done - n;
		while (done < n)
		{
			r <<= 8;
			r |= TF(is.front());
			done += 8;
			if (done <= n)
			{
				is.pop_front();
			}
			else r >>= done - n;
		}
		return r & ((TF(1) << n) - 1);
	}

	TF __forceinline decode_target(TF t)
	{
		r = R / t;
		return std::min(t - 1, D / r);
	}

	void arithmetic_decode(TF l, TF h, TF t)
	{
		// r already set by decode_target
		D = D - r * l;
		if (h < t)
			R = r * (h - l);
		else
			R = R - r * l;

		int s = 0;
		while (R <= QUARTER) {
			R <<= 1;
			s++;
		}
		D = (D << s) | read_n_bits(s);
	}
};


template <typename TF, typename TS>
struct Context : Coder<TF> {
	using Coder<TF>::f;

	std::vector<TF> F;
	std::vector<TF> C;
	TS n;
	int n_bits;

	Context(TS n) : F(n, 0), C(n, 0), n(n)
	{
		n_bits = ((int)floor(log2(n)));
	}

	template <typename It>
	void init(It first, It last)
	{
		TS i = 0;
		for (; first != last; ++first) {
			fenwick_increment_frequency(i++, *first);
		}
	}

	TS __forceinline forward(TS i)
	{
		return i + (i & -i);
	}
	TS __forceinline backward(TS i)
	{
		return i - (i & -i);
	}
	void install_symbol(TS s)
	{
		fenwick_increment_frequency(s, 1);
	}
	void __forceinline fenwick_increment_frequency(TS s, TF inc)
	{
		TS i = s + 1;
		while (i <= n) {
			F[i - 1] += inc;
			i = forward(i);
		}
		C[s] += inc;
	}
	void __forceinline fenwick_decrement_frequency(TS s, TF inc)
	{
		TS i = s + 1;
		while (i <= n) {
			F[i - 1] -= inc;
			i = forward(i);
		}
		C[s] -= inc;
	}
	TF __forceinline fenwick_get_frequency(TS s)
	{
		TS i = s + 1;
		TF h = 0;
		while (i != 0) {
			h += F[i - 1];
			i = backward(i);
		}
		return h;
	}
	TF __forceinline get_t()
	{
		return fenwick_get_frequency(n - 1);
	}
	void __forceinline fenwick_get_range(TS s, TF &low, TF &high)
	{
		high = fenwick_get_frequency(s);
		TF current = C[s];
		low = high - current;
	}
	TS __forceinline fenwick_get_symbol(TF original, TF &low, TF &high)
	{
		TS s = 0;
		TF target = original;
		if (original >= F[0])
		{
			TS mid = 1 << n_bits;
			while (mid > 0) {
				if (s + mid <= n && F[s + mid - 1] <= target) {
					target -= F[s + mid - 1];
					s += mid;
				}
				mid >>= 1;
			}
		}
		low = original - target;
		high = low + C[s];
		return s;
	}

	void halve()
	{
		for (TS i = 0; i < n; ++i) {
			fenwick_decrement_frequency(i, fenwick_get_frequency(i) >> 1);
		}
	}

	void encode(Encoder<TF> &encoder, TS s)
	{
		TF t = get_t();
		if (t > (1 << (f >> 1)))
		{
			halve();
			t = get_t();
		}
		TF l, h;
		fenwick_get_range(s, l, h);
		encoder.arithmetic_encode(l, h, t);

		fenwick_increment_frequency(s, 1);
	}
	TS decode(Decoder<TF> &decoder)
	{
		TF t = get_t();
		if (t > (1 << (f >> 1)))
		{
			halve();
			t = get_t();
		}
		TF target = decoder.decode_target(t);
		TF l, h;
		TS s = fenwick_get_symbol(target, l, h);
		decoder.arithmetic_decode(l, h, t);

		fenwick_increment_frequency(s, 1);

		return s;
	}
};

