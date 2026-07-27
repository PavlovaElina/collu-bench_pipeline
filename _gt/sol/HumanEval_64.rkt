#lang racket
(define (vowels_count s)
  (define cs (string->list s))
  (define base (for/sum ([c cs] #:when (member (char-downcase c) '(#\a #\e #\i #\o #\u))) 1))
  (+ base (if (and (> (length cs) 0) (char=? (char-downcase (last cs)) #\y)) 1 0)))
