#lang racket
(define (skjkasdkd lst)
  (define (prime? x)
    (and (> x 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt x)))]) (not (= 0 (modulo x d))))))
  (define primes (filter prime? lst))
  (if (null? primes) 0
      (let ([p (apply max primes)])
        (for/sum ([c (in-string (number->string p))]) (- (char->integer c) 48)))))
